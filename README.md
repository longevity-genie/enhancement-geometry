# Enhanced geometry generator

This project is an **enhanced geometry generator**: a script that reads values from your enhancement list and turns them into **one-off geometry**—each run can produce a unique shape driven by that data.

## Physical scale (units)

Design within these bounds so sizes stay printable and consistent:

| Axis | Max size (units) | Real-world note |
|------|------------------|-----------------|
| **X / Y** | **150** | Treat as **150 mm** in the plane |
| **Z** | **7.5** | Height limit along Z |

Because the usable width across XY is 150 units, **circle radius is constrained to 20–75** (half of 150 at most).

---

## Parameters you can edit

### Radius (per category)

- Each **radius** maps to **one category** in your enhancement list.
- Derive a numeric value with **two decimal places** between **20** and **75**—for example from selected genes, nucleotides, or any deterministic rule you define so the same inputs yield the same geometry.

### Spacing (between circles)

- **Spacing** is the distance **between** circles. It can stay **equal** for all gaps like it is now, or you can **edit** it.
- **Idea:** use spacing to encode **importance** or **weight** (e.g. of genes or categories).
- **Endpoints:** the **first** and **last** circles stay fixed at the **maximum** span (or at a **set** distance you choose).
- **Interior:** gaps between the other circles are split by **ratios** derived from how “main” or important each segment is. For example, if the value associated with the region between circle 0 and circle 2 makes circle 1 **more important than 2 but less than 0**, circle 1 **moves closer to 0**; you establish the ratios and **divide the available distance** accordingly.

### Seed and number of points

- **Number of points** should stay between **35** and **120**.
- Choose them from user-related input (e.g. a **name** or other fields) so the point count and **seed** feel tied to the person or dataset, while staying in range.

### Extrusion

- Extrusion is kept **negative** by default; you can change it if needed.
- It can stay **uniform** across the model, or vary by region if you later assign different values where the design allows.
- Keep values between **-0.5** and **-0.2**, with **two decimal places**.

---

## Summary limits

| Parameter | Range / rule |
|-----------|----------------|
| Radius | **20.00–75.00** (two decimals), one per category |
| Points | **35–120** (e.g. derived from user input) |
| Extrusion | **-0.50 to -0.20** (two decimals) |
| XY footprint | **≤ 150** units (≈ 150 mm) |
| Z height | **≤ 7.5** units |
