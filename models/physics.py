import torch

class Fossen3DOF:
    def __init__(self,
                 m=30.0, Iz=4.1,  # rigid body
                 X_u=-20.0, Y_v=-40.0, N_r=-5.0,          # linear damping
                 X_uu=-10.0, Y_vv=-20.0, N_rr=-2.0,       # nonlinear damping
                 X_du=5.0, Y_dv=10.0, N_dr=1.0,           # added mass
                 L=0.5):                                   # distance between thrusters
        self.m = m
        self.Iz = Iz
        self.L = L

        # linear damping
        self.X_u = X_u
        self.Y_v = Y_v
        self.N_r = N_r

        # nonlinear damping
        self.X_uu = X_uu
        self.Y_vv = Y_vv
        self.N_rr = N_rr

        # added mass
        self.X_du = X_du
        self.Y_dv = Y_dv
        self.N_dr = N_dr

    def forward(self, u, v, r, tL, tR):
        # Thrust inputs
        X = tL + tR
        Y = torch.zeros_like(X)
        N = (tR - tL) * (self.L / 2.0)

        device = u.device

        # Inverse mass (diagonal, scalar constants)
        M_inv_diag = torch.tensor([
            1.0 / (self.m + self.X_du),
            1.0 / (self.m + self.Y_dv),
            1.0 / (self.Iz + self.N_dr)
        ], dtype=torch.float32, device=device)

        # Linear damping per sample
        D_lin = torch.stack([
            self.X_u * u,
            self.Y_v * v,
            self.N_r * r
        ], dim=-1)  # [batch, 3]

        # Nonlinear damping per sample
        D_nl = torch.stack([
            self.X_uu * u * torch.abs(u),
            self.Y_vv * v * torch.abs(v),
            self.N_rr * r * torch.abs(r)
        ], dim=-1)  # [batch, 3]

        # Combine damping
        D_total = D_lin + D_nl

        # Net forces/moments
        tau = torch.stack([X, Y, N], dim=-1)  # [batch, 3]

        # Compute acceleration (elementwise per dimension)
        accel = (tau - D_total) * M_inv_diag  # broadcasting [batch,3] * [3]

        return accel  # [batch, 3]
