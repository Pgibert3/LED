import pygame
import sys

class VStrip:
    def __init__(self, num_leds, spacing=20, radius=10):
        self.num_leds = num_leds
        self.radius = radius
        self.spacing = spacing

    def start(self):
        x_dim = self.num_leds * (2*self.radius + self.spacing) + self.spacing
        y_dim = 2 * (self.radius + self.spacing)
        self.screen = pygame.display.set_mode((x_dim, y_dim))
        self.screen.fill((255,255,255))
        for i in range(0, self.num_leds):
            x = i * (2*self.radius + self.spacing) + self.radius + self.spacing
            y = self.radius + self.spacing
            black = (0, 0, 0)
            pygame.draw.circle(self.screen, black, (x,y), self.radius, 1)
            self.show()

    def check_closed(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def set_strip(self, leds, wr_black=True):
        white = (255, 255, 255)
        self.screen.fill(white)
        if wr_black:
            for i in range(0, len(leds)):
                color = (leds[i][0], leds[i][1], leds[i][2])
                x = i * (2*self.radius + self.spacing) + self.radius + self.spacing
                y = self.radius + self.spacing
                pygame.draw.circle(self.screen, color, (x,y), self.radius, 0)
        else:
            for i in range(0, len(self.leds)):
                color = (leds[i][0], leds[i][1], leds[i][2])
                pygame.draw.circle(self.screen, color, (x,y), radius, 1)

    def show(self):
        pygame.display.update()
