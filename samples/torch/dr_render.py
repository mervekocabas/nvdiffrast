import torch
from smplx import SMPL
import nvdiffrast.torch as dr
from typing import Optional, Union
from PIL import Image


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
    f = torch.tensor(coord_frame.triangles).to(torch.int32)
    
    R = coord_frame.get_rotation_matrix_from_xyz((torch.pi / 4, 0, 0))
    
    cam_ext = torch.eye(4, device='cuda', dtype=torch.float32)[None]
    # cam_ext[:, :3, :3] = torch.from_numpy(R).cuda().float()[None]
    cam_ext[:, :3, 3] = torch.tensor([0, 0, -2], device='cuda', dtype=torch.float32)[None]
    # import ipdb; ipdb.set_trace()
    renderer = NVDRRenderer(cam_intrinsics=cam_int, init_smpl=True)
    print(renderer.projection_matrix)
    print(renderer.test_cam_ext)
    # img = renderer.forward(vertices=v, faces=f, vertex_colors=vc, cam_ext=cam_ext, return_pil_image=True, test_mode=False, return_rgba=True)
    img = renderer.forward(None, faces=None, cam_ext=cam_ext, return_pil_image=True, test_mode=True, return_rgba=True)
    print(img.size)
    img.save('test.png')