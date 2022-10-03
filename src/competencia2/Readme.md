# Competencia 2

## Enfoque general

Se decidió emplear el 80% del tiempo a mejorar la parametrización del modelo LGBM, y el 20% al feature engineering. En este marco, el objetivo fue crear un modelo lo suficientemente estable que tuviera la menor pérdida de ganancia posible.

## Experimentos explorados en esta competencia

Entre otros:
- Lidiar con el data drifting mediante el dropeo selectivo de variables problemáticas
- Experimentos con parámetro `extra_trees`
- Experimentos con parámetro `bagging_fraction`
- Experimentos con parámetro `feature_fraction`
- Experimentos con `learning_rate` fijo, o de distintos intervalos 
- Elección de submission final en base a búsqueda de "minimos locales" de ganancia pública
- Experimentos con distintos números de iteraciones (300-700)

## Lecciones aprendidas a partir de una caída de 39 puestos en el leaderboard Privado

* El [KaggleHack](https://github.com/dmecoyfin/labo/tree/main/src/KaggleHack) demostraba cómo se distribuían los picos de ganancia, en base a lo cual era recomendable hacer una submission que no bajara de los 9000 envíos. Por no comprender el objetivo del KaggleHack mandé una submission de 6500 envíos.   
![](https://i.imgur.com/gB2ouhu.png)

* Dada mi (baja) experiencia en `data.table` y la falta de exploraciones en profundidad del dataset, 20% del tiempo no fue suficiente para generar feature engineering a conciencia para lidiar con el data drift, por lo que terminé descartando toda modificación al dataset que no fuera la base provista por la materia. Esto impactó en la calidad de los envíos generados y va a agravarse en las próximas competencias.


*  No hacer chanchadas: como tratar de dejar el learning_rate fijo me daba error, lo sumé como hiper parámetro al espacio de búsqueda de la BO con un rango ínfimo, lo que afectó la performance de la BO. La próxima, preguntar en Zulip.

*  (continuará)

## Scripts
* BO: https://github.com/mcnanton/labo/blob/main/src/competencia2/BO_con_FE.R
* Salida final: https://github.com/mcnanton/labo/blob/main/src/competencia2/salida_cortes.R
