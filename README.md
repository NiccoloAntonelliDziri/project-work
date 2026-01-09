## Requirements

- Python 3.13.11

1. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate
```

2. Install required Python packages:

```bash
pip install -r base_requirements.txt
```

3. Build the Cython code:

```bash
python src/setup_ga.py build_ext --inplace
```

4. Run the program:

```bash
python s350296.py
```