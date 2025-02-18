import torch
from smplx import SMPL
from smplx import SMPLX 
import nvdiffrast.torch as dr
from typing import Optional, Union
from PIL import Image
import numpy as np
import os
from collections import defaultdict

class NVDRRenderer():
    def __init__(
        self,
        fx: Optional[Union[torch.Tensor, float]] = None,
        fy: Optional[Union[torch.Tensor, float]] = None,
        cx: Optional[Union[torch.Tensor, float]] = None,
        cy: Optional[Union[torch.Tensor, float]] = None,
        width: Optional[Union[torch.Tensor, int]] = None,
        height: Optional[Union[torch.Tensor, int]] = None,
        cam_intrinsics: Optional[torch.Tensor] = None,
        znear: float = 0.05,
        zfar: float = 100.0,
        faces: Optional[torch.Tensor] = None,
        init_smpl: bool = False,
    ) -> None:
        
        self.device = 'cuda'
        
        self.ctx = dr.RasterizeCudaContext()
        self.znear = znear
        self.zfar = zfar
        self.cam_intrinsics = cam_intrinsics
        
        if cam_intrinsics is None:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.width = width
            self.height = height
        
        if init_smpl:
            self.smpl = SMPL('samples/data/body_models/SMPL_python_v.1.1.0/smpl/models',gender='male').to(self.device)
            self.faces = self.smpl.faces_tensor.to(torch.int32)
            self.test_vertices = self.smpl().vertices
            self.test_cam_ext = torch.eye(4, device=self.device, dtype=torch.float32)[None]
            self.test_cam_ext[:, :3, 3] = torch.tensor([0, 0, -2], device=self.device, dtype=torch.float32)[None]
        
        if faces is not None:
            self.faces = faces
    
    def get_projection_matrix_from_intrinsics(self, cam_intrinsics: torch.Tensor):
        assert len(cam_intrinsics.shape) == 3, "cam_intrinsics must have batch dimension"
        
        self.fx = cam_intrinsics[:, 0, 0]
        self.fy = cam_intrinsics[:, 1, 1]
        self.cx = cam_intrinsics[:, 0, 2]
        self.cy = cam_intrinsics[:, 1, 2]
        self.width = cam_intrinsics[:, 0, 2] * 2
        self.height = cam_intrinsics[:, 1, 2] * 2
        self.width = self.width.to(torch.int32)
        self.height = self.height.to(torch.int32)
        
        return self.get_projection_matrix()

    def get_projection_matrix(self):
        assert self.fx is not None, "fx must be specified"
        assert self.fy is not None, "fy must be specified"
        
        assert self.cx is not None or self.width is not None, "cx or width must be specified"
        assert self.cy is not None or self.height is not None, "cy or height must be specified"
        
        self.cx = self.cx if self.cx is not None else self.width / 2.
        self.cy = self.cy if self.cy is not None else self.height / 2.
        
        # if isinstance(self.fx, float):
        #     self.fx = torch.tensor(self.fx, device=self.device, dtype=torch.float32)[None, None]
        #     self.fy = torch.tensor(self.fy, device=self.device, dtype=torch.float32)[None, None]
        #     self.cx = torch.tensor(self.cx, device=self.device, dtype=torch.float32)[None, None]
        #     self.cy = torch.tensor(self.cy, device=self.device, dtype=torch.float32)[None, None]
        
        B = 1 if isinstance(self.fx, float) else self.fx.shape[0]

        P = torch.zeros((B, 4,4), device=self.device, dtype=torch.float32)

        P[:, 0, 0] = 2.0 * self.fx / self.width
        P[:, 1, 1] = 2.0 * self.fy / self.height
        P[:, 0, 2] = 1.0 - 2.0 * self.cx / self.width
        P[:, 1, 2] = 2.0 * self.cy / self.height - 1.0
        P[:, 3, 2] = -1.0

        if self.zfar is None:
            P[:, 2, 2] = -1.0
            P[:, 2, 3] = -2.0 * self.znear
        else:
            P[:, 2, 2] = (self.zfar + self.znear) / (self.znear - self.zfar)
            P[:, 2, 3] = (2 * self.zfar * self.znear) / (self.znear - self.zfar)

        return P
    
    def cam_intrinsics(self) -> torch.Tensor:
        if self.cam_intrinsics is None:
            B = 1 if isinstance(self.fx, float) else self.fx.shape[0]
            self.cam_intrinsics = torch.zeros((B, 3, 3), device=self.device, dtype=torch.float32)
            self.cam_intrinsics[:, 0, 0] = self.fx
            self.cam_intrinsics[:, 1, 1] = self.fy
            self.cam_intrinsics[:, 0, 2] = self.cx
            self.cam_intrinsics[:, 1, 2] = self.cy
        else:
            return self.cam_intrinsics
    
    @property
    def projection_matrix(self) -> torch.Tensor:
        if self.cam_intrinsics is not None:
            return self.get_projection_matrix_from_intrinsics(self.cam_intrinsics)
        else:
            return self.get_projection_matrix()
            
    def get_mvp_matrix(self, cam_ext: torch.Tensor):
        return torch.matmul(self.projection_matrix, cam_ext)
    
    def transform_pos(self, mtx, pos):
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=-1)
        return torch.matmul(posw, mtx.transpose(-1, -2))
    
    def render_rgba(self, pos, pos_idx, vtx_col, col_idx):
        w = self.width
        h = self.height
        
        if w % 8 != 0:
            w = int((w // 8 + 1) * 8)
            
        if h % 8 != 0:
            h = int((h // 8 + 1) * 8)

        rast_out, _ = dr.rasterize(self.ctx, pos, pos_idx, resolution=[h, w])
        color, _ = dr.interpolate(vtx_col, rast_out, col_idx)
        color = dr.antialias(color, rast_out, pos, pos_idx)

        # alpha = (color.sum(-1) > 0.0).unsqueeze(-1) # torch.clamp(rast_out[..., -1:], 0, 1)
        color = color[:, :self.height, :self.width, :]
        # alpha = alpha[:, :self.height, :self.width, :]
        return color #, alpha

    def forward(
        self,
        vertices: torch.Tensor, 
        cam_ext: torch.Tensor, 
        faces: Optional[torch.Tensor] = None, 
        vertex_colors: Optional[torch.Tensor] = None,
        return_pil_image: bool = False,
        test_mode: bool = False,
        return_rgba: bool = True,
    ) -> torch.Tensor:
        
        if test_mode:
            vertices = self.test_vertices
            if cam_ext is None:
                cam_ext = self.test_cam_ext
            
        if faces is None:
            faces = self.faces
            
        if vertex_colors is None:
            vertex_colors = torch.ones_like(vertices)
            
        if return_rgba:
            vertex_colors = torch.cat([vertex_colors, torch.ones_like(vertex_colors[..., :1])], axis=-1)
        
        vertices = vertices.to(self.device)
        faces = faces.to(self.device)
        faces = faces.to(torch.int32)
        vertex_colors = vertex_colors.to(self.device)
        cam_ext = cam_ext.to(self.device)
            
        assert len(vertices.shape) == 3, "vertices must have batch dimension"
        assert len(cam_ext.shape) == 3, "cam_ext must have batch dimension"
        assert vertices.shape[0] == cam_ext.shape[0], "vertices and cam_ext must have same batch dimension"
        assert len(faces.shape) == 2, "faces must Fx3"
        
        mvp = self.get_mvp_matrix(cam_ext)
        pos_clip = self.transform_pos(mvp, vertices)

        rend_img = self.render_rgba(pos_clip, faces, vertex_colors, faces)
            
        if return_pil_image:
            pil_images = []
            for i in range(rend_img.shape[0]):
                pil_images.append(Image.fromarray((rend_img[i].detach().cpu().numpy() * 255).astype('uint8')))
            
            if len(pil_images) == 1:
                return pil_images[0]
            else:
                return pil_images
        
        return rend_img

    
if __name__ == "__main__":
    """
    # Load BEDLAM Data
    bedlam_data = np.load("samples/data/bedlam_input/filtered_seq_000000.npz", allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert imgname and gender to lists
    imgnames = bedlam_data["imgname"].tolist()
    genders = bedlam_data["gender"].tolist()

    # Group SMPLX data by image
    image_dict = defaultdict(list)

    for i, imgname in enumerate(imgnames):
        image_dict[imgname].append(i)  # Store the index of each person

    # Initialize SMPLX model (we will change gender dynamically)
    smplx_male = SMPLX('samples/data/body_models/smplx/models/smplx/', gender='male').to(device)
    smplx_female = SMPLX('samples/data/body_models/smplx/models/smplx/', gender='female').to(device)
        
    # Loop over unique images
    for imgname, indices in image_dict.items():
        # Define canvas size (e.g., 1920x1080 for BEDLAM)
        width, height = 1920, 1080
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
    
        # Collect all people for this image
        all_vertices, all_faces, all_vertex_colors = [], [], []
        all_cam_int, all_cam_ext = [], [] 

        for i in indices:
            gender = genders[i]
            smplx_model = smplx_female if gender == "female" else smplx_male  # Select model

            # Get SMPLX parameters
            pose = torch.tensor(bedlam_data["pose_world"][i], dtype=torch.float32).to(device)
            shape = torch.tensor(bedlam_data["shape"][i], dtype=torch.float32).to(device)
            pose = pose.unsqueeze(0)
            shape = shape.unsqueeze(0)
            
            # Get per-person camera matrices
            cam_int = torch.tensor(bedlam_data["cam_int"][i], dtype=torch.float32).to(device)
            cam_ext = torch.tensor(bedlam_data["cam_ext"][i], dtype=torch.float32).to(device)
            import ipdb; ipdb.set_trace()

            # Store in batch lists
            all_cam_int.append(cam_int.unsqueeze(0))  
            all_cam_ext.append(cam_ext.unsqueeze(0))

            # Get 3D mesh
            smplx_output = smplx_model(
                body_pose=pose[:,3:66],
                global_orient=pose[:,:3],
                betas=shape[:,:10],
                use_pca=False,
            )
            
            vertices = smplx_output.vertices.squeeze(0)
            faces = smplx_model.faces_tensor.to(torch.int32)

            # Set vertex colors (default: white)
            vertex_colors = torch.ones_like(vertices)

            # Store for batch rendering
            all_vertices.append(vertices)
            all_faces.append(faces)
            all_vertex_colors.append(vertex_colors)

        # Convert lists to tensors
        all_vertices = torch.stack(all_vertices).to(device)
        all_vertex_colors = torch.stack(all_vertex_colors).to(device)
        all_cam_int = torch.stack(all_cam_int).squeeze(1).to(device) 
        all_cam_ext = torch.stack(all_cam_ext).squeeze(1).to(device) 
        batch_size = all_vertices.shape[0]
        
        for i in range(batch_size):
            renderer = NVDRRenderer(cam_intrinsics=all_cam_int[i].unsqueeze(0), faces=faces)
            img = renderer.forward(
                vertices=all_vertices[i].unsqueeze(0),
                faces=all_faces[0],
                vertex_colors=all_vertex_colors[i].unsqueeze(0),
                cam_ext=all_cam_ext[i].unsqueeze(0),
                return_pil_image=True,
                return_rgba=True
            )
            # Paste each rendered person onto the canvas
            canvas.paste(img, (0, 0), img) 
    
        img_output_path = os.path.join('outputs', f"rendered_{imgname}")
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        img.save(img_output_path)
    
    
    """
    """
    # 🔹 Load BEDLAM Data
    bedlam_data = np.load("samples/data/bedlam_input/filtered_seq_000000.npz", allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract imgname
    imgname = bedlam_data["imgname"].tolist()
    gender = bedlam_data["gender"].tolist()
    
    # 🔹 Extract necessary fields
    pose = torch.tensor(bedlam_data["pose_world"], dtype=torch.float32)  # Pose parameters
    shape = torch.tensor(bedlam_data["shape"], dtype=torch.float32)      # SMPL Shape
    cam_int = torch.tensor(bedlam_data["cam_int"], dtype=torch.float32)  # Intrinsics
    cam_ext = torch.tensor(bedlam_data["cam_ext"], dtype=torch.float32)  # Extrinsics
   
    pose = pose.to(device)
    shape = shape.to(device)
    
    # Initialize SMPLX models for both genders
    smplx_male = SMPLX('samples/data/body_models/smplx/models/smplx/', gender='male').to(device)
    smplx_female = SMPLX('samples/data/body_models/smplx/models/smplx/', gender='female').to(device)
    
    output_dir = "outputs"
    
    # Iterate over each sample in the batch
    for i in range(len(imgname)):
        imgnames = imgname[i]
        genders = gender[i]  # Get gender for the current sample
        
        smplx_model = smplx_female if genders == "female" else smplx_male  # Select model

        pose_i = pose[i].unsqueeze(0)  
        shape_i = shape[i].unsqueeze(0)

        smplx_output = smplx_model(
            body_pose=pose_i[:, 3:66], 
            global_orient=pose_i[:, :3], 
            betas=shape_i[:, :10], 
            use_pca=False
        )

        vertices = smplx_output.vertices  # (1, N, 3)
        faces = smplx_model.faces_tensor.to(torch.int32)  # Get faces
        
        vertex_colors = torch.ones_like(vertices)  # (B, N, 3)
        
        renderer = NVDRRenderer(cam_intrinsics=cam_int[i].unsqueeze(0), faces=faces)
        
        # Render each image separately
        img = renderer.forward(
            vertices=vertices, faces=faces, vertex_colors=vertex_colors, 
            cam_ext=cam_ext[i].unsqueeze(0), return_pil_image=True, return_rgba=True
        )
        
        img_path = os.path.join(output_dir, f"rendered_{imgnames}")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # Save the image with the correct filename
        img.save(img_path)
    """
    
    bedlam_data = np.load("samples/data/bedlam_input/filtered_second_human_image.npz", allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🔹 Extract necessary fields
    pose = torch.tensor(bedlam_data["pose_cam"], dtype=torch.float32)  # Pose parameters
    shape = torch.tensor(bedlam_data["shape"], dtype=torch.float32)      # SMPL Shape
    cam_int = torch.tensor(bedlam_data["cam_int"], dtype=torch.float32)  # Intrinsics
    cam_ext = torch.tensor(bedlam_data["cam_ext"], dtype=torch.float32)  # Extrinsics
    cam_int = cam_int.unsqueeze(0)
    #cam_ext = cam_ext.unsqueeze(0)
    pose = pose.unsqueeze(0)
    shape = shape.unsqueeze(0)
     
    #cam_ext[:, 2:3] *= -1
    
    pose = pose.to(device)
    shape = shape.to(device)
    
    # 🔹 Initialize SMPL Model
    smplx = SMPLX('samples/data/body_models/smplx/models/smplx/', gender='female').cuda()
    smplx = smplx.to(device) 
    c_trans = torch.from_numpy(bedlam_data['trans_cam']).to(device)
    # import ipdb; ipdb.set_trace()
    # c_trans[1:] *= -1
    smplx_output = smplx(body_pose=pose[:, 3:66], global_orient=pose[:,:3], betas=shape[:, :10], transl=c_trans[None], use_pca=False )
    vertices = smplx_output.vertices  # (B, N, 3)
    cam_trans = cam_ext[:3, 3].to(device)
    # cam_trans[2] *= -1
    # cam_trans[1] *= -1
    #cam_trans = torch.tensor([0, 0, -6]).to(device)
    vertices = vertices + cam_trans[None, None]
    
    rot_x = torch.tensor(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    ).to(device).float()
    
    rot_z = torch.tensor(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    ).to(device).float()
    
    rot_x = rot_z @ rot_x
    
    vertices = (rot_x[None, None] @ vertices[..., None])[..., 0]
    
    faces = smplx.faces_tensor.to(torch.int32)  # SMPL faces
    
    # 🔹 Prepare vertex colors (white by default)
    vertex_colors = torch.ones_like(vertices)  # (B, N, 3)

    # 🔹 Initialize the Renderer
    renderer = NVDRRenderer(cam_intrinsics=cam_int, faces=faces)

    # 🔹 Render Image
    cam_ext = torch.eye(4)
    cam_ext = cam_ext.unsqueeze(0)
    img = renderer.forward(vertices=vertices, faces=faces, vertex_colors=vertex_colors, cam_ext=cam_ext, return_pil_image=True)

    # 🔹 Save Image
    img.save("outputs/cam_tests.png")
    
    """
    cam_int = torch.zeros(1, 4, 4)
    cam_int[:, 0, 0] = 1004.63
    cam_int[:, 1, 1] = 1004.63
    cam_int[:, 0, 2] = 638.0
    cam_int[:, 1, 2] = 358.5
    
    # renderer = NVDRRenderer(fx=1000., fy=1000., cx=512., cy=512., width=1024, height=1024, init_smpl=True)
    # print(renderer.projection_matrix)
    
    import open3d as o3d
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vc = torch.tensor(coord_frame.vertex_colors).to(torch.float32)[None]
    v = torch.tensor(coord_frame.vertices).to(torch.float32)[None]
    f = torch.tensor(np.array(coord_frame.triangles, dtype=np.int32), dtype=torch.int32)
    
    R = coord_frame.get_rotation_matrix_from_xyz((torch.pi / 4, 0, 0))
    
    cam_ext = torch.eye(4, device='cuda', dtype=torch.float32)[None]
    # cam_ext[:, :3, :3] = torch.from_numpy(R).cuda().float()[None]
    cam_ext[:, :3, 3] = torch.tensor([0, 1, -6], device='cuda', dtype=torch.float32)[None]
    # import ipdb; ipdb.set_trace()
    renderer = NVDRRenderer(cam_intrinsics=cam_int, init_smpl=True)
    print(renderer.projection_matrix)
    print(renderer.test_cam_ext)
    img = renderer.forward(vertices=v, faces=f, vertex_colors=vc, cam_ext=cam_ext, return_pil_image=True, test_mode=False, return_rgba=True)
    #img = renderer.forward(None, faces=None, cam_ext=cam_ext, return_pil_image=True, test_mode=True, return_rgba=True)
    print(img.size)
    img.save('outputs/coordinate_sys.png')
    """