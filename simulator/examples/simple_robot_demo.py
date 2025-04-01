import time

def main():
    # Create environment
    env = ShipEnv()
    
    # Create viewer
    viewer = MujocoViewer(env.sim)
    
    # Set initial camera position
    viewer.cam.distance = 5.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    
    # Add key callback for collider visibility
    def key_callback(keycode):
        if keycode == ord('c'):  # 'c' key to toggle colliders
            env.toggle_collision_visibility()
        elif keycode == ord('s'):  # 's' key to show all colliders
            env.show_all_colliders()
    
    viewer.user_scn_cb = key_callback
    
    # Run simulation
    while viewer.is_running():
        # Step simulation
        env.sim.step()
        
        # Update viewer
        viewer.render()
        
        # Small delay to control simulation speed
        time.sleep(0.01)
    
    viewer.close()

if __name__ == "__main__":
    main() 