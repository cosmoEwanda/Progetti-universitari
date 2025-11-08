import pygame
import random
import time
from Human import Human_1, Human_2, Human_3
# All'inizio del file
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from Utils import Util

unhappy_counts = []
average_tolerances = []
resigned_counts = []
movement_counts = []
flux = []

#proposta di implementazione per preference... non so se vuoi associare ad ogni 
#valore di preference anche un peso
White = (0, 255, 0) # Green
Asian = (255, 255, 0) # Gold
Black = (226, 122, 84) # Red
Mixed = (201, 201, 201) # Ligth grey
Other = (0, 192, 192) #Light blue
ethnic_distribution = {
    'White': {"color": White, "percentage": 0.538, "tolerance": 0.60,
              "preferences": [Asian, Other]},   # Green
    'Asian': {"color": Asian, "percentage":0.208, "tolerance":0.65,
              "preferences": [White, Mixed]}, # Gold
    'Black': {"color": Black, "percentage": 0.135, "tolerance": 0.50,
              "preferences": [Mixed, Other]},# Black
    'Mixed': {"color": Mixed, "percentage":0.057, "tolerance": 0.95,
              "preferences": [White, Asian, Black, Other]},# Purple
    'Other': {"color": Other, "percentage":0.062, "tolerance": 0.55,
              "preferences": [Mixed, Black]}      # light blue
}



# Costanti
ROWS, COLS = 20, 20
CELL_SIZE = 30
WINDOW_SIZE = (COLS * CELL_SIZE, ROWS * CELL_SIZE)
FPS = 10 # Velocità della simulazione

def create_grid(rows, cols, empty_ratio, mode):
    match mode:
        case 0:
            def select(color, *_):
                return Human_1(color=color, tolerance = 0.3)
        case 1:
            def select(color, tolerance, *_):
                return Human_1(color=color, tolerance=tolerance)
        case 2:
            def select(color, tolerance, age, gender, *_):
                return Human_2(color=color, age=age, gender=gender, tolerance=tolerance)
        case 3: 
            def select(color, tolerance, age, gender, preferences):
                return Human_3(color=color, age=age, gender=gender, tolerance=tolerance, preferences=preferences)         
        case _:
            def select(color, tollerance, *_):
                return Human_1(color, tollerance)
             
    grid = []
    total_cells = rows * cols
    num_agents = int(total_cells * (1 - empty_ratio))
    num_empty = total_cells - num_agents

    # Genera la lista di agenti secondo le percentuali
    agents = []
    for group in ethnic_distribution:
        count = int(num_agents * ethnic_distribution[group]["percentage"])
        color = ethnic_distribution[group]["color"]
        tolerance = ethnic_distribution[group]["tolerance"]
        preferences = ethnic_distribution[group]["preferences"]
        for _ in range(count):
            age = random.randint(0, 1)
            gender = random.randint(0, 1)
            agent = select(color, tolerance, age, gender, preferences) #riempie la cella di agenti
            agents.append(agent)
            
    # Aggiungi celle vuote
    agents.extend([Human_1((255, 255, 255)) for _ in range(num_empty)])

    # Mischia l'ordine per distribuzione casuale
    random.shuffle(agents)

    # Costruisci la griglia
    for i in range(rows):
        row = []
        for j in range(cols):
            # Assicurati che ci siano abbastanza agenti per popolare la griglia
            if agents:
                row.append(agents.pop())  # Popola la cella con un agente
            else:
                row.append(Human_1((255, 255, 255)))  # Se finisce la lista, metti una cella vuota
        grid.append(row)

    return grid




def draw_grid(screen, grid, simulation_active):
    
    font = pygame.font.SysFont(None, 36)  # Usa un font di default, dimensione 36
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            agent_ = grid[i][j]
            color = agent_.color
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            
            # Se è Human_2 o Human_3, applica decorazioni
            if isinstance(agent_, Human_2):
                # A strisce se è femmina (gender == 1)
                if agent_.getGender() == 1:
                    stripe_color = tuple(max(0, c - 60) for c in color)  # schiarisce un po’
                    for k in range(0, CELL_SIZE, 4):
                        pygame.draw.line(screen, stripe_color, (j * CELL_SIZE, i * CELL_SIZE + k),
                                         (j * CELL_SIZE + CELL_SIZE, i * CELL_SIZE + k), 3)

                # Bordo più spesso se anziano (age == 0)
                if agent_.getAge() == 0:
                    pygame.draw.rect(screen, (0, 0, 0), rect, 3)  # bordo nero spesso
                else:
                    pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # bordo normale
            else:
                # Per Human_1, bordo normale
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            

            if (simulation_active and agent_.color != agent_.empty and agent_.wanna_move):
                text = font.render("*", True, (0, 0, 0))
                text_rect = text.get_rect(center=(
                    j * CELL_SIZE + CELL_SIZE // 2,  # x = colonna
                    i * CELL_SIZE + CELL_SIZE // 2   # y = riga
                ))
                screen.blit(text, text_rect)
          
            """
                
            # Disegna l'asterisco se l'agente è "rassegnato" dopo 700 iterazioni
            if hasattr(agent_, "getAsterisk") and agent_.getAsterisk():
                text = font.render("*", True, (0, 0, 0))
                text_rect = text.get_rect(center=(
                    j * CELL_SIZE + CELL_SIZE // 2,  # x = colonna
                    i * CELL_SIZE + CELL_SIZE // 2   # y = riga
                ))
                screen.blit(text, text_rect)
            """
    
    mid_row = len(grid) // 2
    mid_col = len(grid[0]) // 2
    
    #disegno le righe dei quartieri
    pygame.draw.line(screen, (0,0,220),(mid_col*CELL_SIZE,0),(mid_col*CELL_SIZE,screen.get_height()),5)
    pygame.draw.line(screen, (0,0,220),(0,mid_row*CELL_SIZE),(screen.get_width(), mid_row*CELL_SIZE),5)
    
            
def simulate_step(grid):

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            agent = grid[i][j]
            if agent.color == (255, 255, 255):
                continue
            neighbours = Util.get_moore_neighbourhood(grid, i, j)
            agent.compareNeighbourhood(neighbours)
            
    # Trova agenti che vogliono muoversi
    movers = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j].wanna_move]
    empty = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j].color == (255, 255, 255)]

    random.shuffle(movers)
    random.shuffle(empty)

    for (i, j), (ni, nj) in zip(movers, empty):
        grid[ni][nj], grid[i][j] = grid[i][j], Human_1((255, 255, 255))
        
        grid[i][j].movements+=1
        grid[ni][nj].movements+=1

def first_model(mode):
    empty_ratio = 0.9
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Schelling Simulation")
    clock = pygame.time.Clock()
    # Font per il testo
    

    grid = create_grid(ROWS, COLS, empty_ratio, mode=0)
    quarters = Util.divide_into_quarters(grid)
    Nagents = Util.calculate_Nagents(quarters)
    
    

    running = True
    frame_count = 0  # Conta il numero di fotogrammi passati
    simulation_active = True
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if(simulation_active):
            # Simula un passo
            simulate_step(grid)

            # Raccoglie dati a ogni passo
            num_agents = sum(1 for row in grid for a in row if a.color != a.empty)
            num_unhappy = sum(a.getWanna_move() for row in grid for a in row if a.color != a.empty)
            unhappy_counts.append(num_unhappy / num_agents)

            avg_tolerance = sum(a.getTolerance() for row in grid for a in row if a.color != a.empty) / num_agents
            average_tolerances.append(avg_tolerance)

            num_resigned = sum(a.getAsterisk() for row in grid for a in row if hasattr(a, "getAsterisk") and a.color != a.empty)
            resigned_counts.append(num_resigned)

            total_movements = sum(a.movements for row in grid for a in row if a.color != a.empty)
            movement_counts.append(total_movements / num_agents)  # movimenti medi per agente
            
            quarters = Util.divide_into_quarters(grid)
            
            tmp = Util.calculate_Nagents(quarters)
            flux.append(np.mean(np.abs(tmp - Nagents)))
            Nagents = tmp
            
            
            frame_count += 1
            #if frame_count % (FPS * 5) == 0: 
            if False:
                for i in range(len(grid)):
                    for j in range(len(grid[0])):
                        agent = grid[i][j]
                        agent.updateTolerance()
                        #print(f"actual tolerance={agent.tolerance}")
            
            # Termina dopo 3 minuti (1800 frame)
            if frame_count >= 12 * FPS:
                print("Simulazione terminata dopo 3 minuti.")
                simulation_active = False
                
                
              
                total_cells = len(grid)*len(grid[0])
                full_theil = Util.calculate_Full_Theil_Index(ethnic_distribution, int(total_cells * (1 - empty_ratio)), quarters)
                local_theil = Util.calculate_Local_Theil_Index(ethnic_distribution, int(total_cells * (1 - empty_ratio)), quarters)
                plot_simulation_data()
                #plot_ethnic_distribution_by_quarter(quarters, ethnic_distribution)

            # Disegna la griglia
            draw_grid(screen, grid, not simulation_active)
        

        pygame.display.flip()
        clock.tick(FPS)
            
    pygame.quit()
    

def moving_average(data, window_size):
    if window_size % 2 == 0:  # finestra pari
        pad_left = window_size // 2
        pad_right = window_size // 2 - 1
    else:
        pad_left = pad_right = window_size // 2

    padded = np.pad(data, (pad_left, pad_right), mode='edge')
    return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')




def moving_median(data, window_size=5):
    return median_filter(data, size=window_size, mode='nearest')

    

def plot_ethnic_distribution_by_quarter(quarters, ethnic_distribution):
    race_names = list(ethnic_distribution.keys())
    num_quarters = len(quarters)

    # Conta la distribuzione per quartiere
    data = []
    for q in range(num_quarters):
        count_per_race = {race: 0 for race in race_names}
        for row in quarters[q]:
            for agent in row:
                for race, props in ethnic_distribution.items():
                    if agent.color == props["color"]:
                        count_per_race[race] += 1
        data.append([count_per_race[race] for race in race_names])
    
    # Transponi per plottare: ogni razza ha 4 valori (uno per quartiere)
    data = list(zip(*data))  # 5 liste, ciascuna con 4 elementi

    bar_width = 0.2
    x = range(len(race_names))

    plt.figure(figsize=(10, 6))
    for i in range(num_quarters):
        plt.bar([xi + i * bar_width for xi in x],
                data[i],
                width=bar_width,
                label=f'Quartiere {i+1}')

    plt.xticks([xi + 1.5 * bar_width for xi in x], race_names)
    plt.xlabel('Razza')
    plt.ylabel('Numero di agenti')
    plt.title('Distribuzione etnica nei 4 quartieri')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_simulation_data():
    steps = list(range(len(unhappy_counts)))

    plt.figure(figsize=(12, 10))

    # Grafico 1: infelicità
    plt.subplot(2, 2, 1)
    plt.plot(steps, unhappy_counts, label='Infelicità (prop. agenti che vogliono muoversi)', color='red')
    plt.xlabel("Step")
    plt.ylabel("Percentuale")
    plt.title("Infelicità nel tempo")
    plt.grid(True)

    # Grafico 2: tolleranza media
    plt.subplot(2, 2, 2)
    plt.plot(steps, average_tolerances, label='Tolleranza media', color='blue')
    plt.xlabel("Step")
    plt.ylabel("Tolleranza")
    plt.title("Tolleranza media nel tempo")
    plt.grid(True)

    # Grafico 3: movimenti
    plt.subplot(2, 2, 3)
    plt.plot(steps, movement_counts, label='Movimenti medi per agente', color='green')
    plt.xlabel("Step")
    plt.ylabel("Movimenti medi")
    plt.title("Movimenti degli agenti")
    plt.grid(True)

    # Grafico 4: flusso
    plt.subplot(2, 2, 4)
    
    window_size = 100  # Puoi regolarlo a piacere

    flux_array = np.array(flux)

    # Versioni filtrate
    ma_flux = moving_average(flux_array, window_size)
    #med_flux = moving_median(flux_array, window_size)

    # Allineamento degli step (la media mobile accorcia il segnale!)
    #valid_steps = steps[len(steps) - len(ma_flux):]
    
    #smoothed_steps = steps[len(steps) - len(smoothed_flux):]  # per allineare
    plt.plot(steps, flux, label="Flusso originale", color="orange", alpha=0.4)
    plt.plot(steps, ma_flux, label="Media mobile", color="blue", linewidth=2)
    #plt.plot(steps, med_flux, label="Mediana mobile", color="green", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Flusso")
    plt.title("Flusso migratorio medio")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()



first_model(mode = 1)