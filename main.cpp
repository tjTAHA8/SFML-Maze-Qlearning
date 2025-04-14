#include <SFML/Graphics.hpp>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>

using namespace std;
using namespace sf;

const int CELL_SIZE = 15;
const int GRID_ROWS = 12;
const int GRID_COLS = 20;

const Color BACKGROUND_COLOR(15, 15, 30);
const Color FIRST_VISIT(50, 50, 100);
const Color SECOND_VISIT(225, 225, 225);
const Color WALL_COLOR(20, 20, 40);
const Color START_COLOR(0, 255, 0);
const Color GOAL_COLOR(255, 0, 0);
const Color AGENT_COLOR(255, 173, 0);
const Color PATH_COLOR(255, 100, 0);

struct Cell {
    int visitCount = 0;
    bool walls[4] = { true, true, true, true };
    int x, y;

    Cell(int x, int y) : x(x), y(y) {}
};

const int dx[4] = { 0, 1, 0, -1 };
const int dy[4] = { -1, 0, 1, 0 };

vector<vector<Cell>> grid;
vector<Cell*> frontier;
Cell* current = nullptr;
vector<Cell*> optimalPath;

// Q-Learning variables
bool mazeGenerated = false;
bool qLearningStarted = false;
Cell* agentCurrent = nullptr;
Cell* startCell = nullptr;
Cell* goalCell = nullptr;

double epsilon = 0.1;
double alpha = 0.1;
double gamma = 0.9;

vector<vector<vector<double>>> qTable(GRID_ROWS, vector<vector<double>>(GRID_COLS, vector<double>(4, 0.0)));

void initializeGrid() {
    grid.clear();
    for (int y = 0; y < GRID_ROWS; y++) {
        vector<Cell> row;
        for (int x = 0; x < GRID_COLS; x++) {
            row.emplace_back(x, y);
        }
        grid.push_back(row);
    }

    int midY = GRID_ROWS / 2;
    startCell = &grid[midY][0];
    goalCell = &grid[midY][GRID_COLS - 1];
    startCell->walls[1] = false;
    goalCell->walls[3] = false;

    // Prim's algorithm 
    int centerX = GRID_COLS / 2;
    int centerY = GRID_ROWS / 2;
    grid[centerY][centerX].visitCount = 1;
    frontier.push_back(&grid[centerY][centerX]);
    agentCurrent = startCell;
}

void removeWalls(Cell& a, Cell& b) {
    int xDiff = b.x - a.x;
    int yDiff = b.y - a.y;

    if (xDiff == 1) {
        a.walls[1] = false;
        b.walls[3] = false;
    }
    else if (xDiff == -1) {
        a.walls[3] = false;
        b.walls[1] = false;
    }
    else if (yDiff == 1) {
        a.walls[2] = false;
        b.walls[0] = false;
    }
    else if (yDiff == -1) {
        a.walls[0] = false;
        b.walls[2] = false;
    }
}

void computeOptimalPath() {
    optimalPath.clear();
    Cell* current = startCell;
    int steps = 0;

    while (current != goalCell && steps < GRID_ROWS * GRID_COLS) {
        optimalPath.push_back(current);
        auto& qValues = qTable[current->y][current->x];
        int action = distance(qValues.begin(), max_element(qValues.begin(), qValues.end()));

        int newX = current->x;
        int newY = current->y;
        bool canMove = false;

        switch (action) {
        case 0: if (!current->walls[0]) { newY--; canMove = true; } break;
        case 1: if (!current->walls[1]) { newX++; canMove = true; } break;
        case 2: if (!current->walls[2]) { newY++; canMove = true; } break;
        case 3: if (!current->walls[3]) { newX--; canMove = true; } break;
        }

        if (newX < 0 || newX >= GRID_COLS || newY < 0 || newY >= GRID_ROWS) canMove = false;

        if (canMove) current = &grid[newY][newX];
        else break;

        steps++;
    }

    if (current == goalCell) optimalPath.push_back(current);
}

int main() {
    mt19937 rng(time(nullptr));
    RenderWindow window(
        VideoMode(GRID_COLS * CELL_SIZE, GRID_ROWS * CELL_SIZE),
        "Maze Generation with Q-Learning"
    );
    initializeGrid();

    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        if (!mazeGenerated) {
            if (!frontier.empty()) {
                uniform_int_distribution<int> dist(0, frontier.size() - 1);
                int index = dist(rng);
                current = frontier[index];
                frontier.erase(frontier.begin() + index);

                current->visitCount = 2;

                vector<Cell*> neighbors;
                for (int i = 0; i < 4; i++) {
                    int nx = current->x + dx[i];
                    int ny = current->y + dy[i];
                    if (nx >= 0 && nx < GRID_COLS && ny >= 0 && ny < GRID_ROWS) {
                        if (grid[ny][nx].visitCount == 2) {
                            neighbors.push_back(&grid[ny][nx]);
                        }
                    }
                }

                if (!neighbors.empty()) {
                    uniform_int_distribution<int> ndist(0, neighbors.size() - 1);
                    Cell* neighbor = neighbors[ndist(rng)];
                    removeWalls(*current, *neighbor);
                }

                for (int i = 0; i < 4; i++) {
                    int nx = current->x + dx[i];
                    int ny = current->y + dy[i];
                    if (nx >= 0 && nx < GRID_COLS && ny >= 0 && ny < GRID_ROWS) {
                        Cell& neighbor = grid[ny][nx];
                        if (neighbor.visitCount == 0) {
                            neighbor.visitCount = 1;
                            frontier.push_back(&neighbor);
                        }
                    }
                }
            }
            else {
                mazeGenerated = true;
            }
        }
        else if (qLearningStarted) {
            // Q-Learning 
            uniform_real_distribution<double> rand(0.0, 1.0);
            double random_value = rand(rng);
            int action;

            if (random_value < epsilon) {
                uniform_int_distribution<int> action_dist(0, 3);
                action = action_dist(rng);
            }
            else {
                int x = agentCurrent->x;
                int y = agentCurrent->y;
                auto& qValues = qTable[y][x];
                action = distance(qValues.begin(), max_element(qValues.begin(), qValues.end()));
            }

            bool canMove = false;
            int newX = agentCurrent->x;
            int newY = agentCurrent->y;

            switch (action) {
            case 0:
                if (!agentCurrent->walls[0]) { newY--; canMove = true; }
                break;
            case 1:
                if (!agentCurrent->walls[1]) { newX++; canMove = true; }
                break;
            case 2:
                if (!agentCurrent->walls[2]) { newY++; canMove = true; }
                break;
            case 3:
                if (!agentCurrent->walls[3]) { newX--; canMove = true; }
                break;
            }

            if (newX < 0 || newX >= GRID_COLS || newY < 0 || newY >= GRID_ROWS) canMove = false;

            double reward;
            if (newX == goalCell->x && newY == goalCell->y) {
                reward = 100.0;
                computeOptimalPath();
            }
            else if (!canMove) {
                reward = -10.0;
            }
            else {
                reward = -1.0;
            }

            Cell* nextCell = canMove ? &grid[newY][newX] : agentCurrent;

            int prevY = agentCurrent->y;
            int prevX = agentCurrent->x;
            auto& qValues = qTable[prevY][prevX];
            double currentQ = qValues[action];
            double maxNextQ = *max_element(qTable[nextCell->y][nextCell->x].begin(), qTable[nextCell->y][nextCell->x].end());
            qValues[action] = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);

            agentCurrent = nextCell;

            if (agentCurrent == goalCell) {
                agentCurrent = startCell;
            }
        }
        else {
            qLearningStarted = true;
        }

        window.clear(BACKGROUND_COLOR);

        for (int y = 0; y < GRID_ROWS; y++) {
            for (int x = 0; x < GRID_COLS; x++) {
                Cell& cell = grid[y][x];
                Color cellColor;

                if (&cell == startCell) {
                    cellColor = START_COLOR;
                }
                else if (&cell == goalCell) {
                    cellColor = GOAL_COLOR;
                }
                else {
                    switch (cell.visitCount) {
                    case 0: cellColor = BACKGROUND_COLOR; break;
                    case 1: cellColor = FIRST_VISIT; break;
                    case 2: cellColor = SECOND_VISIT; break;
                    }
                }

                RectangleShape rect(Vector2f(CELL_SIZE, CELL_SIZE));
                rect.setPosition(x * CELL_SIZE, y * CELL_SIZE);
                rect.setFillColor(cellColor);
                window.draw(rect);

                const float wallThickness = 2.0f;
                if (cell.walls[0]) {
                    RectangleShape wall(Vector2f(CELL_SIZE, wallThickness));
                    wall.setPosition(x * CELL_SIZE, y * CELL_SIZE);
                    wall.setFillColor(WALL_COLOR);
                    window.draw(wall);
                }
                if (cell.walls[1]) {
                    RectangleShape wall(Vector2f(wallThickness, CELL_SIZE));
                    wall.setPosition((x + 1) * CELL_SIZE - wallThickness, y * CELL_SIZE);
                    wall.setFillColor(WALL_COLOR);
                    window.draw(wall);
                }
                if (cell.walls[2]) {
                    RectangleShape wall(Vector2f(CELL_SIZE, wallThickness));
                    wall.setPosition(x * CELL_SIZE, (y + 1) * CELL_SIZE - wallThickness);
                    wall.setFillColor(WALL_COLOR);
                    window.draw(wall);
                }
                if (cell.walls[3]) {
                    RectangleShape wall(Vector2f(wallThickness, CELL_SIZE));
                    wall.setPosition(x * CELL_SIZE, y * CELL_SIZE);
                    wall.setFillColor(WALL_COLOR);
                    window.draw(wall);
                }
            }
        }

        // Draw optimal path 
        for (Cell* cell : optimalPath) {
            if (cell == startCell || cell == goalCell) continue;

            RectangleShape pathRect(Vector2f(CELL_SIZE, CELL_SIZE));
            pathRect.setPosition(cell->x * CELL_SIZE, cell->y * CELL_SIZE);
            pathRect.setFillColor(PATH_COLOR);
            window.draw(pathRect);
        }

        if (qLearningStarted && agentCurrent) {
            // Draw Pac-Man shape
            float centerX = agentCurrent->x * CELL_SIZE + CELL_SIZE / 2.0f;
            float centerY = agentCurrent->y * CELL_SIZE + CELL_SIZE / 2.0f;
            float radius = CELL_SIZE / 2.0f;
            float mouthAngle = 30.0f; // Mouth opening angle

            VertexArray pacman(TriangleFan);
            pacman.append(Vertex(Vector2f(centerX, centerY), AGENT_COLOR)); // Center

            // Create mouth opening
            for (float angle = mouthAngle / 2; angle <= 360.0f - mouthAngle / 2; angle += 1.0f) {
                float rad = angle * 3.14159f / 180.0f;
                pacman.append(Vertex(Vector2f(
                    centerX + radius * cos(rad),
                    centerY + radius * sin(rad)
                ), AGENT_COLOR));
            }

            window.draw(pacman);
        }

        window.display();
        sleep(milliseconds(10));
    }

    return 0;
}