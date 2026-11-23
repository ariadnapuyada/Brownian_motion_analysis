import pygame
import pymunk
import pymunk.pygame_util
import random

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FPS = 60
SOLVENT_COUNT = 500
SOLVENT_RADIUS = 4
BROWNIAN_RADIUS = 25
INITIAL_IMPULSE = 100  # Sets the "Temperature"

def create_particle(space, pos, radius, mass, color):
    # 1. Define physical properties
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    
    # 2. Define the shape for collisions
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 1.0  # Perfectly elastic (no energy loss)
    shape.friction = 0.0
    shape.color = color
    
    space.add(body, shape)
    return body

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Initialize Physics Space
    space = pymunk.Space()
    space.gravity = (0, 0) # Zero gravity for fluid simulation

    # Create Walls (Static bodies)
    static_lines = [
        pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 5),
        pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 5),
        pymunk.Segment(space.static_body, (WIDTH, HEIGHT), (0, HEIGHT), 5),
        pymunk.Segment(space.static_body, (0, HEIGHT), (0, 0), 5)
    ]
    for line in static_lines:
        line.elasticity = 1.0
        space.add(line)

    # Create the Large Brownian Particle
    brownian_body = create_particle(space, (WIDTH/2, HEIGHT/2), BROWNIAN_RADIUS, 20, (255, 0, 0, 255))
    
    # Create Small Solvent Particles
    for _ in range(SOLVENT_COUNT):
        x = random.randint(50, WIDTH-50)
        y = random.randint(50, HEIGHT-50)
        body = create_particle(space, (x, y), SOLVENT_RADIUS, 1, (100, 100, 255, 255))
        
        # Give them random initial movement (Thermal Energy)
        impulse = (random.uniform(-INITIAL_IMPULSE, INITIAL_IMPULSE), 
                   random.uniform(-INITIAL_IMPULSE, INITIAL_IMPULSE))
        body.apply_impulse_at_local_point(impulse)

    # Simulation Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((20, 20, 20)) # Dark background
        
        # Step the physics forward
        space.step(1/FPS)
        
        # Draw everything
        space.debug_draw(draw_options)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()