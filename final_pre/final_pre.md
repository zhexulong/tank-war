---
marp: true
theme: default
paginate: true
---

# Agent for Tank Battle

---

# Simplified Rules

- A tank can perform exactly one action per turn:
  - Move forward, Turn left, Turn right, Fire forward, Stay still

- All bricks are destructible
- One-shot kill the tank
---

# Logic Agent Strategy

1. **Aligning to the Enemy**

    If not on same row/column as enemy, compute horizontal and vertical offset, Face direction with smaller absolute offset and move forward.


2. **Destroying Obstructing Bricks**
    If next forward position is blocked, fire to destroy the brick

3. **Firing When Aligned**

    When on same row/column as enemy, turn to face enemy and fire continuously until it is eliminated or moves out of alignment

---

# Advantages

- **Straightforward and effective rules, high Win Rate vs. Unskilled Opponents**

- **No Training Required, No data collection needed and Immediate reaction**

- **Interpretable Expert Strategy for latter Agent Training**


# Disadvantages

- **No Bullet Dodging, vulnerable to skilled opponents**

- **Cannot Predict Enemy Movement, only reacts to current state**
