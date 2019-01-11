import csv
from datetime import datetime

# Constantes, directorios y archivos a procesar.
DIR = "datos/"
FILE_COORDENADAS = "DistancesCoordenadasUTM.csv"
FILE_CALIBRACION = "DatosCalibracion.csv"
FILE_RECEPTORES = ["DatosRC1.csv", "DatosRC2.csv", "DatosD1.csv", "DatosD2.csv"] 
##FILE_RECEPTORES = ["DatosRC1_10k.csv", "DatosRC2_10k.csv", "DatosD1_10k.csv", "DatosD2.csv"]  # dev #

NUMREC = 4   # cantidad de receptores

# - - - - 

# Parche para solucionar problemas de formato de fechas:
def arreglar_fecha(f):
  if f=='1/26/2018': return '26/01/2018'
  if f=='1/27/2018': return '27/01/2018'
  if f=='1/28/2018': return '28/01/2018'
  if f=='1/29/2018': return '29/01/2018'
  if f=='2/18/2018': return '18/02/2018'
  if f=='2/19/2018': return '19/02/2018'
  if f=='2/20/2018': return '20/02/2018'
  if f=='2/21/2018': return '21/02/2018'
  if f=='2/22/2018': return '22/02/2018'
  if f=='2/23/2018': return '23/02/2018'
  if f=='2/24/2018': return '24/02/2018'
  if f=='2/25/2018': return '25/02/2018'
  if f=='6/2/2018':  return '02/06/2018'
  if f=='6/3/2018':  return '03/06/2018'
  if f=='6/8/2018':  return '08/06/2018'
  if f=='6/9/2018':  return '09/06/2018'
  return f

# - - - - - - 

# Leo los datos de las coordenadas de los puntos, y los cargo en un diccionario. 
# Ejemplo:  coordenadasUTM{90} --> {'X': 463965.4513, 'Y': 6109294.1496}
# significa que el punto 90 está en esas coordenadas espaciales.
coordenadasUTM = {}
with open(DIR + FILE_COORDENADAS, 'r') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=',')
  for row in reader:
    coordenadasUTM[int(row['Punto'])] = {'X':float(row['X']), 'Y':float(row['Y'])}

# - - - - - - 
# Leo los datos de los puntos de calibración, convirtiendo las fechas 
# y tiempos a timestamps.
puntos_calibracion = []
with open(DIR + FILE_CALIBRACION, 'r') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=',')
  for row in reader:
    if row['Fecha']!="NA":  # ignoro los datos faltantes (puntos sin señal).
      row['Punto'] = int(row['Punto'])
      row['Tag'] = int(row['Tag'])
      row['Fecha'] = arreglar_fecha(row['Fecha'])
      row['timestamp_inicio'] = datetime.strptime(row['Fecha']+' '+ row['Inicio'], '%d/%m/%Y %H:%M:%S')
      row['timestamp_fin'] = datetime.strptime(row['Fecha']+' '+ row['Fin'], '%d/%m/%Y %H:%M:%S')
      row['X'] = coordenadasUTM[row['Punto']]['X']
      row['Y'] = coordenadasUTM[row['Punto']]['Y']
      puntos_calibracion.append(row)

# - - - - - - 

# Leo los datos de las mediciones de los NUMREC receptores, convirtiendo 
# las fechas y tiempos a timestamps.
mediciones = [[] for _ in range(NUMREC)] # Empiezo con NUMREC listas vacías.
for receptor in range(NUMREC):
  with open(DIR + FILE_RECEPTORES[receptor], 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
      if row['Date']!="Date":  # ignoro los encabezados que hay en medio de los datos.
        row['Tag ID'] = int(row['Tag ID'])
        row['Date'] = arreglar_fecha(row['Date'])
        row['timestamp'] = datetime.strptime(row['Date']+' '+ row['Time'], '%d/%m/%Y %H:%M:%S')
        mediciones[receptor].append(row)

# - - - - - - 

## Busco un punto de calibracion para probar, con fecha en junio.
#punto = [p for p in puntos_calibracion if p["Punto"]==339][1]
#print(punto)

#print(coordenadasUTM[punto['Punto']])

## Busco las mediciones correspondientes a ese punto de calibracion.
#meds = [r['Power'] for r in mediciones[3] 
#                  if r['timestamp'] >= punto['timestamp_inicio'] 
#                  and r['timestamp'] <= punto['timestamp_fin'] 
#                  and r['Tag ID'] == punto['Tag']]
#print(meds)
#print(len(meds))            

