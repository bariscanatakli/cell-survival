import pygame
import sys
import math  # Add missing math import
import numpy as np
import time
from environment.species import SpeciesType, Species

class GameVisualizer:
    def __init__(self, world_width, world_height, window_width=1200, window_height=900):
        pygame.init()
        self.world_width = world_width
        self.world_height = world_height
        self.window_width = window_width
        self.window_height = window_height
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Cell Survival Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        
        # Define colors with more vibrant hues for better visualization
        self.colors = {
            'background': (0, 10, 25),  # Darker blue background for contrast
            'grid': (30, 30, 60),  # Slightly more visible grid
            'food': (0, 240, 0),  # Brighter green
            'decay': (230, 140, 0),  # More vibrant orange
            'hazard': (240, 20, 20),  # Brighter red
            SpeciesType.PREDATOR: (220, 0, 0),  # Bright red
            SpeciesType.PREY: (0, 180, 255),  # Bright blue
            SpeciesType.GATHERER: (220, 180, 0),  # Gold
            SpeciesType.SCAVENGER: (180, 0, 220),  # Purple
            'text': (255, 255, 255),  # White
            'exploration': (255, 255, 0, 128),  # Semi-transparent yellow for exploration
            'depletion': (255, 0, 0, 128),  # Semi-transparent red for depletion
            'energy': (0, 255, 50)  # Bright green for energy
        }
        
        # For panning and zooming
        self.view_x = world_width / 2
        self.view_y = world_height / 2
        self.zoom = 0.6  # Start with slightly higher zoom level for better visibility
        self.first_render = True
        self.show_heatmaps = False

        # Rendering optimization flags
        self.use_gpu_acceleration = False
        self.try_enable_gpu_acceleration()
        self.frame_skip = 0  # Initialize with no frame skipping
        self.last_render_time = 0
        self.target_fps = 60  # Set target to 60 FPS
        self.last_cell_count = 0
        
        # Animation parameters for more dynamic visuals
        self.animation_phase = 0
        self.trail_effect = True  # Enable motion trails for better movement visibility
        self.motion_blur = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        
        # Display FPS counter
        self.show_fps = True
        self.fps_history = []

    def try_enable_gpu_acceleration(self):
        """Try to enable GPU acceleration for PyGame if available"""
        # Check if we can use hardware acceleration
        try:
            if hasattr(pygame, 'HWSURFACE'):
                self.screen = pygame.display.set_mode(
                    (self.window_width, self.window_height), 
                    pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
                self.use_gpu_acceleration = True
                print("GPU-accelerated rendering enabled")
            
            # For SDL2 platforms, try to use OpenGL acceleration
            elif hasattr(pygame, 'OPENGL'):
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
                self.screen = pygame.display.set_mode(
                    (self.window_width, self.window_height), 
                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
                self.use_gpu_acceleration = True
                print("OpenGL-accelerated rendering enabled")
        except Exception as e:
            print(f"Could not enable GPU acceleration: {e}")
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height), 
                pygame.RESIZABLE
            )
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            # Save current window size before going fullscreen
            self.old_width, self.old_height = self.window_width, self.window_height
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.window_width, self.window_height = self.screen.get_size()
        else:
            # Restore previous window size
            self.window_width, self.window_height = self.old_width, self.old_height
            self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
    
    def handle_events(self):
        """Handle pygame events and return whether to quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                if not self.is_fullscreen:  # Only adjust if not in fullscreen mode
                    self.window_width, self.window_height = event.size
                    self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                # Handle keyboard controls
                if event.key == pygame.K_ESCAPE:
                    if self.is_fullscreen:
                        self.toggle_fullscreen()  # Exit fullscreen with ESC
                    else:
                        pygame.quit()
                        sys.exit()
                elif event.key == pygame.K_LEFT:
                    self.view_x -= 50
                elif event.key == pygame.K_RIGHT:
                    self.view_x += 50
                elif event.key == pygame.K_UP:
                    self.view_y -= 50
                elif event.key == pygame.K_DOWN:
                    self.view_y += 50
                # Alternative zoom controls (z and x keys)
                elif event.key == pygame.K_z:  # Zoom in with 'z' key
                    self.zoom *= 1.2
                elif event.key == pygame.K_x:  # Zoom out with 'x' key
                    self.zoom /= 1.2
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom *= 1.1
                elif event.key == pygame.K_MINUS:
                    self.zoom /= 1.1
                # Reset view with 'r' key
                elif event.key == pygame.K_r:
                    self.reset_view()
                # Toggle fullscreen with 'f' key
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                # Toggle heatmaps with 'h' key
                elif event.key == pygame.K_h:
                    self.show_heatmaps = not self.show_heatmaps
    
    def reset_view(self):
        """Reset view to center of the world with default zoom"""
        self.view_x = self.world_width // 2
        self.view_y = self.world_height // 2
        self.zoom = 0.5
    
    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        x = int((pos[0] - self.view_x) * self.zoom + self.window_width/2)
        y = int((pos[1] - self.view_y) * self.zoom + self.window_height/2)
        return (x, y)
    
    def adapt_rendering_quality(self, cell_count):
        """Adapt rendering quality based on number of cells and performance to maintain 60fps"""
        import time
        current_time = time.time()
        
        # If last render was less than 1/60th of a second ago and we have more cells,
        # consider skipping frames or reducing detail
        if self.last_render_time > 0:
            render_interval = current_time - self.last_render_time
            fps = 1 / render_interval if render_interval > 0 else 60
            
            # Store FPS history for averaging
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:  # Keep last 30 frames for averaging
                self.fps_history.pop(0)
            
            # Dynamically adjust for 60 FPS target
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else fps
            
            # More aggressive adaptation for consistent 60 FPS
            if avg_fps < 55 and cell_count > 100:  # If below 55 FPS, increase skip
                self.frame_skip = min(self.frame_skip + 1, 3)  # Max of 3 frames skip
            elif avg_fps > 58:  # If close to target, reduce skip
                self.frame_skip = max(self.frame_skip - 1, 0)
        
        self.last_render_time = current_time
        self.last_cell_count = cell_count
        return self.frame_skip > 0

    def render(self, cells, foods, hazards, episode, step, world=None):
        """Render the current state of the simulation with optimizations for 60fps"""
        # Update animation phase for dynamic effects
        self.animation_phase = (self.animation_phase + 0.1) % 360
        
        # Check if we should skip this frame for performance
        if self.adapt_rendering_quality(len(cells)) and step % (self.frame_skip + 1) != 0:
            return

        # Process events for responsive UI
        self.handle_events()
        
        # Apply slight motion blur for smoother appearance if trails are enabled
        if self.trail_effect and len(cells) < 300:  # Disable for large simulations
            self.screen.fill(self.colors['background'])
            # Add a slight trail effect by keeping previous frame with transparency
            self.motion_blur.fill((0, 0, 0, 10))  # Very slight persistence
            self.screen.blit(self.motion_blur, (0, 0))
        else:
            self.screen.fill(self.colors['background'])
        
        # Center on cells on first render
        if self.first_render and cells:
            # Find average cell position
            avg_x = sum(cell.position[0] for cell in cells) / len(cells)
            avg_y = sum(cell.position[1] for cell in cells) / len(cells)
            self.view_x = avg_x
            self.view_y = avg_y
            self.first_render = False
        
        # Only draw grid when zoomed in enough to see it clearly and not too many cells
        if self.zoom > 0.3 and len(cells) < 500:
            # Optimize grid drawing - only draw every other line when zoomed out
            grid_step = 1 if self.zoom > 0.8 else (2 if self.zoom > 0.5 else 4)
            grid_size = 50 * self.zoom
            grid_offset_x = int(self.view_x * self.zoom) % int(grid_size)
            grid_offset_y = int(self.view_y * self.zoom) % int(grid_size)
            
            # Draw horizontal grid lines
            for y in range(-grid_offset_y, self.window_height, int(grid_size) * grid_step):
                pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.window_width, y))
                
            # Draw vertical grid lines
            for x in range(-grid_offset_x, self.window_width, int(grid_size) * grid_step):
                pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.window_height))
        
        # Draw heatmaps only if enabled, world is provided, and not too many cells (performance)
        if self.show_heatmaps and world is not None and len(cells) < 300:
            self.draw_heatmaps(world)
        
        # Adaptive rendering based on entity count and world size
        simplified_rendering = len(cells) > 250 or len(foods) > 800  # Adjusted thresholds
        ultra_simplified = len(cells) > 800 or len(foods) > 2000
        
        # Draw foods with optimized batching
        if ultra_simplified:
            # Just draw a few representative foods for very large simulations
            sample_size = min(400, len(foods))
            if sample_size > 0:
                sample_step = len(foods) // sample_size
                sampled_foods = [foods[i] for i in range(0, len(foods), sample_step)]
                # Draw as simple pixels in a batch
                food_positions = [self.world_to_screen(food.position) for food in sampled_foods]
                food_positions = [(x, y) for x, y in food_positions if 0 <= x < self.window_width and 0 <= y < self.window_height]
                if food_positions:
                    # Use a surface and draw all at once
                    for pos in food_positions:
                        pygame.draw.circle(self.screen, (0, 255, 0), pos, 1)
        elif simplified_rendering:
            # Batch processing for better performance
            food_positions = [(food.position[0], food.position[1]) for food in foods]
            if food_positions:
                # Create a surface just once and blit food items
                visible_foods = []
                for pos in food_positions:
                    screen_pos = self.world_to_screen(pos)
                    if (0 <= screen_pos[0] < self.window_width and 
                        0 <= screen_pos[1] < self.window_height):
                        visible_foods.append(screen_pos)
                
                # Draw all visible foods at once using points if there are many
                if len(visible_foods) > 400:
                    pygame.draw.polygon(self.screen, (0, 255, 0), visible_foods, 1)
                else:
                    for pos in visible_foods:
                        pygame.draw.circle(self.screen, (0, 255, 0), pos, 2)
        else:
            # Regular food rendering for normal scenarios with pulsing effect
            visible_count = 0
            for food in foods:
                screen_pos = self.world_to_screen(food.position)
                if (0 <= screen_pos[0] < self.window_width and 
                    0 <= screen_pos[1] < self.window_height):
                    # Limit the number of drawn items for performance
                    visible_count += 1
                    if visible_count > 800:  # Cap at 800 visible items
                        break
                        
                    # Regular food is green, decaying food is yellow-orange based on remaining energy
                    if hasattr(food, 'is_decay') and food.is_decay:
                        # Calculate color based on remaining energy (fade from yellow to red)
                        decay_ratio = food.energy_value / 15.0  # Assuming max energy is 15
                        color = (255, int(255 * decay_ratio), 0)
                        radius = max(4 * self.zoom * decay_ratio, 2)
                    else:
                        color = self.colors['food']
                        # Add pulsing effect for food
                        pulse = abs(math.sin(self.animation_phase / 10 + hash(str(food.position)) % 10))
                        radius = max(3 * self.zoom + pulse, 2)
                    
                    pygame.draw.circle(self.screen, color, screen_pos, radius)
        
        # Draw hazards - simplified for performance
        if ultra_simplified:
            # Only draw a few hazards when there are many cells
            sample_size = min(100, len(hazards))
            if sample_size > 0 and len(hazards) > 0:
                sample_step = max(1, len(hazards) // sample_size)
                for i in range(0, len(hazards), sample_step):
                    if i < len(hazards):
                        hazard = hazards[i]
                        screen_pos = self.world_to_screen(hazard.position)
                        if (0 <= screen_pos[0] < self.window_width and 
                            0 <= screen_pos[1] < self.window_height):
                            size = 2  # Minimal size for ultra simplified mode
                            pygame.draw.rect(self.screen, self.colors['hazard'], 
                                        (screen_pos[0]-size, screen_pos[1]-size, size*2, size*2))
        else:
            # Regular hazard rendering with danger pulsing
            for hazard in hazards:
                screen_pos = self.world_to_screen(hazard.position)
                if (0 <= screen_pos[0] < self.window_width and 
                    0 <= screen_pos[1] < self.window_height):
                    # Add subtle pulsing effect to hazards
                    pulse = abs(math.sin(self.animation_phase / 8 + hash(str(hazard.position)) % 10))
                    size = max(5 * self.zoom + pulse, 3)
                    pygame.draw.rect(self.screen, self.colors['hazard'], 
                                (screen_pos[0]-size, screen_pos[1]-size, size*2, size*2))
        
        # Draw cells with adaptive detail level
        species_counts = {species: 0 for species in SpeciesType}
        
        # For very large simulations, sample the cells
        cells_to_render = cells
        if len(cells) > 800:
            sample_size = 500  # Render at most 500 cells
            sample_step = len(cells) // sample_size
            cells_to_render = [cells[i] for i in range(0, len(cells), sample_step)]
        
        # Keep track of all species even if not rendering all cells
        for cell in cells:
            species_counts[cell.species.type] = species_counts.get(cell.species.type, 0) + 1
        
        # Render visible cells
        for cell in cells_to_render:
            screen_pos = self.world_to_screen(cell.position)
            if (0 <= screen_pos[0] < self.window_width and 
                0 <= screen_pos[1] < self.window_height):
                
                # In ultra simplified mode, skip color calculations
                if ultra_simplified:
                    color = self.colors[cell.species.type]
                    radius = max(4 * self.zoom, 2) if not ultra_simplified else 2
                else:
                    # Show starvation effect with color intensity
                    base_color = self.colors[cell.species.type]
                    starvation_factor = min(1.0, getattr(cell, 'last_food_time', 0) / 
                                          (getattr(cell, 'starvation_threshold', 100) * 2))
                    # Make cell appear pale/weak as starvation increases
                    color = (
                        min(255, int(base_color[0] + 50 * starvation_factor)),
                        max(0, int(base_color[1] - 50 * starvation_factor)),
                        max(0, int(base_color[2] - 50 * starvation_factor))
                    )
                    
                    # Add subtle size variation based on energy if not ultra simplified
                    energy_ratio = getattr(cell, 'energy_level', 50) / getattr(cell.species, 'max_energy', 100)
                    radius = max((5 + energy_ratio * 2) * self.zoom, 3)
                
                pygame.draw.circle(self.screen, color, screen_pos, radius)
                
                # Skip energy bars in ultra simplified mode for performance
                if not ultra_simplified and not simplified_rendering:
                    # Draw energy level indicator
                    energy_ratio = getattr(cell, 'energy_level', 50) / getattr(cell.species, 'max_energy', 100)
                    energy_width = max(10 * self.zoom, 5)  # Ensure minimum visibility
                    indicator_height = max(2 * self.zoom, 1)  # Ensure minimum visibility
                    
                    # Draw energy bar background
                    pygame.draw.rect(self.screen, (255, 255, 255),
                                   (screen_pos[0]-energy_width/2, 
                                    screen_pos[1]-radius-indicator_height-1, 
                                    energy_width, indicator_height), 1)
                    
                    # Draw energy bar fill
                    pygame.draw.rect(self.screen, self.colors['energy'],
                                   (screen_pos[0]-energy_width/2, 
                                    screen_pos[1]-radius-indicator_height-1, 
                                    energy_width*energy_ratio, indicator_height))
        
        # Draw stats - with scaling for different window sizes
        info_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Cells: {len(cells)}",
            f"Foods: {len(foods)}",
            f"Hazards: {len(hazards)}",
            f"Zoom: {self.zoom:.2f}x"
        ]
        
        # Add species counts
        for species, count in species_counts.items():
            info_text.append(f"{species.name}: {count}")
        
        # Add FPS display
        if self.show_fps and self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            info_text.append(f"FPS: {avg_fps:.1f}")
        
        # Scale font size based on window size
        font_size = max(16, min(24, int(self.window_height / 40)))
        info_font = pygame.font.Font(None, font_size)
        
        # Create a semi-transparent background for stats
        stats_surface = pygame.Surface((220, len(info_text) * (font_size + 2) + 10), pygame.SRCALPHA)
        stats_surface.fill((0, 0, 30, 180))  # Semi-transparent dark blue
        self.screen.blit(stats_surface, (10, 10))
        
        # Render stats text
        for i, text in enumerate(info_text):
            text_surf = info_font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surf, (15, 15 + i * (font_size + 2)))
        
        # Help text at the bottom - only show in normal mode
        if not ultra_simplified:
            help_font = pygame.font.Font(None, max(14, int(font_size * 0.8)))
            help_text = "Arrows: Move  Z/X: Zoom  R: Reset view  F: Fullscreen  H: Toggle heatmaps  ESC: Exit"
            help_surf = help_font.render(help_text, True, (200, 200, 200))
            
            # Create a semi-transparent background for help text
            help_bg_width = help_surf.get_width() + 20
            help_bg = pygame.Surface((help_bg_width, help_surf.get_height() + 10), pygame.SRCALPHA)
            help_bg.fill((0, 0, 30, 180))  # Semi-transparent dark blue
            
            self.screen.blit(help_bg, (10, self.window_height - help_surf.get_height() - 15))
            self.screen.blit(help_surf, (20, self.window_height - help_surf.get_height() - 10))
        
        # Always target 60 FPS
        pygame.display.flip()
        self.clock.tick(60)
    
    def draw_heatmaps(self, world):
        """Draw heatmaps for exploration and resource depletion"""
        # Create surface for exploration heatmap
        exploration_surface = pygame.Surface((world.width, world.height), pygame.SRCALPHA)
        depletion_surface = pygame.Surface((world.width, world.height), pygame.SRCALPHA)
        
        # Draw exploration levels
        for x in range(0, world.width, 8):  # Skip pixels for performance
            for y in range(0, world.height, 8):
                # Exploration heatmap (yellow)
                if world.exploration_bonus_grid[x, y] > 0.01:
                    intensity = min(128, int(world.exploration_bonus_grid[x, y] * 255))
                    exploration_surface.set_at((x, y), (255, 255, 0, intensity))
                
                # Resource depletion heatmap (red)
                if world.resource_depletion_grid[x, y] > 0.01:
                    intensity = min(128, int(world.resource_depletion_grid[x, y] * 255))
                    depletion_surface.set_at((x, y), (255, 0, 0, intensity))
        
        # Scale and position the heatmap according to the current view
        scaled_exploration = pygame.transform.scale(
            exploration_surface, 
            (int(world.width * self.zoom), int(world.height * self.zoom))
        )
        scaled_depletion = pygame.transform.scale(
            depletion_surface, 
            (int(world.width * self.zoom), int(world.height * self.zoom))
        )
        
        # Calculate position on screen (adjusted by view position)
        pos_x = -int(self.view_x * self.zoom)
        pos_y = -int(self.view_y * self.zoom)
        
        # Blit the heatmaps onto the screen
        self.screen.blit(scaled_exploration, (pos_x, pos_y))
        self.screen.blit(scaled_depletion, (pos_x, pos_y))