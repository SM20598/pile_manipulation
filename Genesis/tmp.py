import torch


n_required = torch.ceil(dist / (speed * self._scene.dt)).max().int().item()
n_current = 0
reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
abort = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
done = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)  # reached_goal | abort

# Track the lifted z-position per env once it finishes
# We'll move finished envs upward by incrementally updating their z each step
lift_steps = 100
lift_z_per_step = self._lift_height[0, 2] / lift_steps  # scalar, same for all envs
lift_progress = torch.zeros(self._n_envs, device=gs.device)  # how many lift steps taken

while not done.all():
    n_current += 1

    # Build per-env position command: finished envs get their current pos locked
    # (they are being lifted), active envs get the sweep fix_pose
    cur_pos = self.plate.get_pos()  # [n_envs, 3]

    # For envs that just finished or are lifting, override fix_pose z upward
    active = ~done  # envs still sweeping

    # Step the scene (all envs advance together)
    self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)

    # Override z for done envs: lift them incrementally
    if done.any():
        lifting_envs = done & (lift_progress < lift_steps)
        if lifting_envs.any():
            lifted_z = (p_end[done, 2] + lift_progress[done] * lift_z_per_step)
            # set_dofs_position per-env for z (dof 2)
            new_z_pose = fix_pose.clone()
            new_z_pose[done, 0] = lifted_z  # dof 2 is index 0 in fix_dofs [2,3,4,5]
            self.plate.set_dofs_position(new_z_pose, dofs_idx_local=fix_dofs)
            lift_progress[done] += 1

    self._scene.step()

    cur_pos = self.plate.get_pos()
    cur_dist = torch.linalg.norm(cur_pos - p_end, axis=1)

    newly_done = (cur_dist < 0.002) & ~done
    reached_goal |= newly_done
    abort |= (n_current > n_required * 1.7) & ~reached_goal & ~done
    done = reached_goal | abort