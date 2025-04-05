import pygame
import sys
import numpy as np
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
        
        # Define colors
        self.colors = {
            'background': (0, 0, 30),
            'grid': (30, 30, 50),
            'food': (0, 200, 0),  # Green
            'decay': (150, 75, 0),  # Brown
            'hazard': (200, 0, 0),  # Red
            SpeciesType.PREDATOR: (255, 100, 100),  # Red
            SpeciesType.PREY: (100, 100, 255),  # Blue
            SpeciesType.GATHERER: (100, 255, 100),  # Green
            SpeciesType.SCAVENGER: (255, 255, 100),  # Yellow
            'text': (255, 255, 255),  # White
            'exploration': (255, 255, 0, 128),  # Semi-transparent yellow for exploration
            'depletion': (255, 0, 0, 128),  # Semi-transparent red for depletion
        }
        
        # Keep track of view position (for scrolling)
        self.view_x = world_width // 2  # Start centered
        self.view_y = world_height // 2  # Start centered
        self.zoom = 0.5  # Start with a reasonable zoom level
        self.first_render = True  # Flag to center on cells on first render
        self.show_heatmaps = False  # Toggle for showing heatmaps

        # Rendering optimization flags
        self.use_gpu_acceleration = False
        self.try_enable_gpu_acceleration()
        self.frame_skip = 0  # For dynamic frame skipping
        self.last_render_time = 0
        self.target_fps = 30
        self.last_cell_count = 0  # For adaptive rendering
    
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
        """Adapt rendering quality based on number of cells and performance"""
        import time
        current_time = time.time()
        
        # If last render was less than 1/30th of a second ago and we have more cells,
        # consider skipping frames or reducing detail
        if self.last_render_time > 0:
            render_interval = current_time - self.last_render_time
            fps = 1 / render_interval if render_interval > 0 else 60
            
            # If FPS drops below target with many cells, increase frame skip
            if fps < self.target_fps * 0.8 and cell_count > 100:
                self.frame_skip = min(self.frame_skip + 1, 5)
            elif fps > self.target_fps * 1.2:
                self.frame_skip = max(self.frame_skip - 1, 0)
        
        self.last_render_time = current_time
        self.last_cell_count = cell_count
        return self.frame_skip > 0

    def render(self, cells, foods, hazards, episode, step, world=None):
        """Render the current state of the simulation"""
        # Check if we should skip this frame for performance
        if self.adapt_rendering_quality(len(cells)) and step % (self.frame_skip + 1) != 0:
            return

        self.handle_events()
        self.screen.fill(self.colors['background'])
        
        # Center on cells on first render
        if self.first_render and cells:
            # Find average cell position
            avg_x = sum(cell.position[0] for cell in cells) / len(cells)
            avg_y = sum(cell.position[1] for cell in cells) / len(cells)
            self.view_x = avg_x
            self.view_y = avg_y
            self.first_render = False
        
        # Draw a grid (optional)
        grid_size = 50 * self.zoom
        grid_offset_x = int(self.view_x * self.zoom) % int(grid_size)
        grid_offset_y = int(self.view_y * self.zoom) % int(grid_size)
        
        # Draw horizontal grid lines
        for y in range(-grid_offset_y, self.window_height, int(grid_size)):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.window_width, y))
            
        # Draw vertical grid lines
        for x in range(-grid_offset_x, self.window_width, int(grid_size)):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.window_height))
        
        # Draw heatmaps if enabled and world is provided
        if self.show_heatmaps and world is not None:
            self.draw_heatmaps(world)
        
        # If world is large and we have many cells, use simplified rendering
        simplified_rendering = len(cells) > 500 or (world and world.width > 4096)
        
        # Draw foods with simplified method if needed
        if simplified_rendering:
            food_positions = [(food.position[0], food.position[1]) for food in foods]
            if food_positions:
                # Create a surface just once and blit food items
                food_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                for pos in food_positions:
                    screen_pos = self.world_to_screen(pos)
                    if (0 <= screen_pos[0] < self.window_width and 
                        0 <= screen_pos[1] < self.window_height):
                        pygame.draw.circle(food_surface, (0, 255, 0), screen_pos, 2)
                self.screen.blit(food_surface, (0, 0))
        else:
            # Draw foods - only those within view
            for food in foods:
                screen_pos = self.world_to_screen(food.position)
                if (0 <= screen_pos[0] < self.window_width and 
                    0 <= screen_pos[1] < self.window_height):
                    # Regular food is green, decaying food is yellow-orange based on remaining energy
                    if food.is_decay:
                        # Calculate color based on remaining energy (fade from yellow to red)
                        decay_ratio = food.energy_value / 15.0  # Assuming max energy is 15
                        color = (255, int(255 * decay_ratio), 0)
                        radius = max(4 * self.zoom * decay_ratio, 2)
                    else:
                        color = (0, 255, 0)
                        radius = max(4 * self.zoom, 2)
                    pygame.draw.circle(self.screen, color, screen_pos, radius)
        
        # Draw hazards - only those within view
        for hazard in hazards:
            screen_pos = self.world_to_screen(hazard.position)
            if (0 <= screen_pos[0] < self.window_width and 
                0 <= screen_pos[1] < self.window_height):
                size = max(5 * self.zoom, 2)
                pygame.draw.rect(self.screen, self.colors['hazard'], 
                               (screen_pos[0]-size, screen_pos[1]-size, size*2, size*2))
        
        # Draw cells - only those within view
        species_counts = {species: 0 for species in SpeciesType}
        for cell in cells:
            screen_pos = self.world_to_screen(cell.position)
            if (0 <= screen_pos[0] < self.window_width and 
                0 <= screen_pos[1] < self.window_height):
                # Draw cell body
                radius = max(6 * self.zoom, 3)  # Ensure minimum visibility
                
                # Show starvation effect with color intensity
                base_color = self.colors[cell.species.type]
                starvation_factor = min(1.0, cell.last_food_time / (cell.starvation_threshold * 2))
                # Make cell appear pale/weak as starvation increases
                color = (
                    min(255, int(base_color[0] + 50 * starvation_factor)),
                    max(0, int(base_color[1] - 50 * starvation_factor)),
                    max(0, int(base_color[2] - 50 * starvation_factor))
                )
                
                pygame.draw.circle(self.screen, color, screen_pos, radius)
                
                # Draw energy level indicator
                energy_ratio = cell.energy_level / cell.species.max_energy
                energy_width = max(10 * self.zoom, 5)  # Ensure minimum visibility
                indicator_height = max(2 * self.zoom, 1)  # Ensure minimum visibility
                
                # Draw energy bar background
                pygame.draw.rect(self.screen, (255, 255, 255),
                               (screen_pos[0]-energy_width/2, 
                                screen_pos[1]-radius-indicator_height-1, 
                                energy_width, indicator_height), 1)
                
                # Draw energy bar fill
                pygame.draw.rect(self.screen, (0, 255, 0),
                               (screen_pos[0]-energy_width/2, 
                                screen_pos[1]-radius-indicator_height-1, 
                                energy_width*energy_ratio, indicator_height))
            
            # Update species count
            species_counts[cell.species.type] = species_counts.get(cell.species.type, 0) + 1
        
        # Draw stats - with scaling for different window sizes
        info_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Cells: {len(cells)}",
            f"Foods: {len(foods)}",
            f"Hazards: {len(hazards)}",
            f"Zoom: {self.zoom:.2f}x",
            f"View: ({int(self.view_x)}, {int(self.view_y)})"
        ]
        
        # Add species counts
        for species, count in species_counts.items():
            info_text.append(f"{species.name}: {count}")
        
        # Scale font size based on window size
        font_size = max(16, min(24, int(self.window_height / 40)))
        info_font = pygame.font.Font(None, font_size)
        
        # Create a semi-transparent background for stats
        stats_surface = pygame.Surface((220, len(info_text) * (font_size + 2) + 10), pygame.SRCALPHA)
        stats_surface.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(stats_surface, (10, 10))
        
        # Render stats text
        for i, text in enumerate(info_text):
            text_surf = info_font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surf, (15, 15 + i * (font_size + 2)))
        
        # Help text at the bottom
        help_font = pygame.font.Font(None, max(14, int(font_size * 0.8)))
        help_text = "Arrows: Move  Z/X: Zoom  R: Reset view  F: Fullscreen  H: Toggle heatmaps  ESC: Exit"
        help_surf = help_font.render(help_text, True, (200, 200, 200))
        
        # Create a semi-transparent background for help text
        help_bg_width = help_surf.get_width() + 20
        help_bg = pygame.Surface((help_bg_width, help_surf.get_height() + 10), pygame.SRCALPHA)
        help_bg.fill((0, 0, 0, 128))  # Semi-transparent black
        
        self.screen.blit(help_bg, (10, self.window_height - help_surf.get_height() - 15))
        self.screen.blit(help_surf, (20, self.window_height - help_surf.get_height() - 10))
        
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