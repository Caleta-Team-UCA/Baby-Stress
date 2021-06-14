# Baby-Stress
## Configuración del entorno
```bash
conda env create -f env.yml
conda activate baby-stress
pip install -r requirements.txt
```

Si se instalan librerías nuevas hay que ejecutar:
```
pip-compile --extra=dev
```