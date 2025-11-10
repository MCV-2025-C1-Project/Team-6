# MCV Module 1 Project

<a href="https://docs.google.com/presentation/d/1MLkC8YCv9SGt4bGrH2qg6LXw_gfLsUtzjOKPDtNkpUo/edit?usp=sharing">Link</a> to final presentation.

## Team 6 Members
- Diego Hernández Antón
- Jordi Ventosa Altimira
- Shinto Machado Furuichi
- Alvaro Javier Díaz Laureano
- Oriol Juan Sabater

## Quick Setup

Clone the repository:

```bash
git clone git@github.com:MCV-2025-C1-Project/Team-6.git
cd Team-6
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate team6-env
```

<p>If you need it, install <a href="https://www.anaconda.com/docs/getting-started/miniconda/install">Miniconda</a>.</p>

## Project Structure

This is the project structure, where `WeekX/` folder contains the code developed during that week.  
Inside each week's folder, there is a `README.md` file explaining the scripts.  
Note that the paths to the Development and Test sets for each week can be specified as arguments in the code,  but by default they are assumed to be located inside the corresponding `WeekX/` folder as `qsdY_wX/` and `qstY_wX/`.

```bash
Team-6/
├── BBDD/
├── .gitignore
├── environment.yml
├── README.md
├── WeekX/
│   ├── qsdY_wX/              # Development set Y for Week X
│   ├── qstY_wX/              # Test set Y for Week X
│   ├── src/                  # Code for Week X
│   └── README.md             # Explanation for Week X code
```
