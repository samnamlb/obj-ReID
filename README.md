
# COMP560 Course Project — Object Re-Identification

This repository contains our group’s work for the **COMP560 Object Re-Identification (ReID) course project**.

The goal of the project is to implement a **ReID model** that maps images of objects (animals, vehicles, people, etc.) into discriminative embeddings, enabling retrieval of the same individual across different viewpoints, cameras, and conditions.

---
## GROUP INSTALL INSTRUCTIONS
---

## Environment Setup

We use a **local virtual environment** for isolation and reproducibility.

### 1. Create the virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

You should now see `(.venv)` in your terminal prompt.

---

## Dependencies

Dependencies are managed with **uv**.


```bash
python -m uv sync
```


---

## Tinker API Setup

This project includes an **optional sanity-check script** that verifies connectivity to the **Tinker** API.

### Create a `.env` file
```bash
notepad .env
```

Add:
```env
TINKER_API_KEY=your_api_key_here
```

### **!!!Never commit `.env`**!!!



