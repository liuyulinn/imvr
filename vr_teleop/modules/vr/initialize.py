def create_viewer(env, name, env_cfg):
    scene = env.get_cpu_scene()

    if name == "bunnyvisionpro":
        from .bunny_visionpro import BunnyVisionProViewer

        env.render()
        viewer = env.viewer

        bunny_viewer = BunnyVisionProViewer(visualize=True, viewer=viewer)
        return bunny_viewer

    elif name == "viewer":
        env.render()
        viewer = env.viewer

        return viewer

    elif name == "anyteleop":
        from .anyteleop import AnyTeleopViewer

        env.render()
        viewer = env.viewer
        anyteleop_viewer = AnyTeleopViewer(visualize=True, viewer=viewer)
        return anyteleop_viewer

    elif "vr" in name:
        from .vr_hand_viewer import VRHandViewer
        from .vr_viewer import VRViewer

        viewer = VRHandViewer() if "hand" in name else VRViewer()
        viewer.set_scene(scene)

        init_vr_root_pose = env_cfg.root_pose.to_sapien_pose()  # type: ignore
        viewer.root_pose = init_vr_root_pose
        return viewer

    else:
        raise NotImplementedError
