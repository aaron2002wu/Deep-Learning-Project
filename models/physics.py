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

    def forward(self, nu, tau):
        u, v, r = nu.T
        X, Y, N = tau.T

        # inertia matrix including added mass
        M = torch.stack([
            [self.m + self.X_du, torch.zeros_like(u), torch.zeros_like(u)],
            [torch.zeros_like(u), self.m + self.Y_dv, torch.zeros_like(u)],
            [torch.zeros_like(u), torch.zeros_like(u), self.Iz + self.N_dr]
        ], dim=0)

        # Linear damping
        D_lin = torch.stack([
            self.X_u * u,
            self.Y_v * v,
            self.N_r * r
        ], dim=1)

        # Nonlinear damping
        D_nonlin = torch.stack([
            self.X_uu * u * torch.abs(u),
            self.Y_vv * v * torch.abs(v),
            self.N_rr * r * torch.abs(r)
        ], dim=1)

        D = D_lin + D_nonlin

        # Simple planar Coriolis approximation (can improve with full matrix)
        # For now we just ignore cross terms
        C = torch.zeros_like(D)

        tau_vec = torch.stack([X, Y, N], dim=1)
        
        # accelerations
        acc = torch.linalg.solve(M, tau_vec - D - C)
        return acc
