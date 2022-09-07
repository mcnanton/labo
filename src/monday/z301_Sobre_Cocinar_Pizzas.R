##
## Sobre como cocinar Pizzas
##
## ---------------------------
## Step 1: Cargando los datos y las librerías
## ---------------------------
##
## Success is a lousy teacher. It seduces smart people into thinking they can't 
## lose.
## --- Bill Gates

# Limpiamos el entorno
rm(list = ls())
gc(verbose = FALSE)

# Librerías necesarias
require("data.table")
require("rpart")
require("ROCR")
require("ggplot2")
require("lubridate")
require("lhs")
require("DiceKriging")
require("mlrMBO")

# Poner la carpeta de la materia de SU computadora local
setwd("C:/Users/PC/Documents/DMEyF")
# Poner sus semillas
semillas <- c(100043, 100049, 100153, 100169, 100183)

# Cargamos el dataset
dataset <- fread("./datasets/competencia1_2022.csv")

# Nos quedamos solo con el 202101
dataset <- dataset[foto_mes == 202101]
# Creamos una clase binaria
dataset[, clase_binaria := ifelse(
                            clase_ternaria == "BAJA+2",
                                "evento",
                                "noevento"
                            )]
# Borramos el target viejo
dataset[, clase_ternaria := NULL]

# Seteamos nuestra primera semilla
set.seed(semillas[1])

# Particionamos de forma estratificada
in_training <- caret::createDataPartition(dataset$clase_binaria,
                     p = 0.70, list = FALSE)
dtrain  <-  dataset[in_training, ]
dtest   <-  dataset[-in_training, ]

## ---------------------------
## Step 2: Nuestra pizza: Un modelo
## ---------------------------

# Calculamos cuanto tarda un modelo "promedio" entrenar.
start_time <- Sys.time()
modelo <- rpart(clase_binaria ~ .,
                data = dtrain,
                xval = 0,
                cp = 0,
                minsplit = 20,
                minbucket = 10,
                maxdepth = 10)
pred_testing <- predict(modelo, dtest, type = "prob")
end_time <- Sys.time()
model_time <- end_time - start_time
print("Tiempo de ajuste en train y predict en test")
print(model_time)

ganancia <- function(probabilidades, clase) {
  return(sum(
    (probabilidades >= 0.025) * ifelse(clase == "evento", 78000, -2000))
  )
}

print("La ganancia NORMALIZADA de nuestro modelo es:")
print(ganancia(pred_testing[, "evento"], dtest$clase_binaria) / 0.3)

## Preguntas
## - ¿Es acaso este el mejor modelo posible?
## - ¿Dónde lo buscamos el mejor modelo?
## - ¿Qué parámetros conoce para un árbol de decisión?
## - ¿Qué espacios de búsqueda tienen los parámetros *maxdepth* y *minsplit*?
## - ¿Cómo se imagina la interacción entre esto dos últimos parámetros?

## ---------------------------
## Step 3: There Ain't No Such Thing As A Free Lunch
## ---------------------------

# Supongamos que sólo vamos a buscar sobre los parámetros *maxdepth* y
# *minsplit*

# Tamaño del espacio de búsqueda de *maxdepth*
n_md <- 30 - 4
# Tamaño del espacio de búsqueda de *minsplit*
n_ms <- 200 - 2
# Cantidad de semillas
n_seeds <- 5

# Estimación de cuanto tardaría en buscar el mejor modelo con 2 parámetros.
print(seconds_to_period(n_md * n_ms * n_seeds * model_time))

# Tamaño del espacio de búsqueda de *minsplit*
n_mb <- 100 - 2

# Estimación de cuanto tardaría en buscar el mejor modelo con 3 parámetros.
print(seconds_to_period(n_md * n_ms * n_seeds * model_time * n_mb))

## Preguntas
## - ¿Dispone del tiempo para realizar esta búsqueda?
## - ¿Qué hacemos cuándo un parámetro tiene valores continuos?

## ---------------------------
## Step 4: Empezando a probar con menos casos
## ---------------------------

set.seed(semillas[1])
dist_uni <- matrix(runif(20), 10, 2)

# LHS Latin hypercube sampling
set.seed(semillas[1])
dist_lhs <- optimumLHS(10, 2)

par(mfrow = c(1, 2))
plot(dist_uni)
plot(dist_lhs)


## Preguntas
## - ¿Cuál distribución considera mejor? Justifique


## ---------------------------
## Step 5: Tomando una muestra de sangre
## ---------------------------

# Armamos una función para modelar con el fin de simplificar el código futuro
modelo_rpart <- function(train, test, cp =  0, ms = 20, mb = 1, md = 10) {
    modelo <- rpart(clase_binaria ~ ., data = train,
                    xval = 0,
                    cp = cp,
                    minsplit = ms,
                    minbucket = mb,
                    maxdepth = md)

    test_prediccion <- predict(modelo, test, type = "prob")
    
    ### Reemplazamos métrica AUC x gcia
    #roc_pred <-  ROCR::prediction(test_prediccion[, "evento"],
    #                test$clase_binaria,
    #                              label.ordering = c("noevento", "evento"))
    #auc_t <-  ROCR::performance(roc_pred, "auc")

    #unlist(auc_t@y.values)
    
    ### Ganancia
    ganancia(test_prediccion[, "evento"], test$clase_binaria) / 0.3
}

# Función para tomar un muestra dejando todos los elementos de la clase BAJA+2
tomar_muestra <- function(datos, resto = 10000) {
      t <- datos$clase_binaria == "evento"
      r <- rep(FALSE, length(datos$clase_binaria))
      r[!t][sample.int(resto, n = (length(t) - sum(t)))] <- TRUE
      t | r
}

set.seed(semillas[1])
ds_sample <- tomar_muestra(dataset)
table(dataset[ds_sample]$clase_binaria)

## Preguntas
## - ¿Qué tipo de muestre se tomó?
## - ¿Hay mejores formas de muestrear?
## - ¿Es bueno muestrear?
## - ¿Qué efectos en las métricas va a producir el muestreo?
## - ¿Por qué se eligió usar el AUC? Se usa con modelos rebalanceados, no se ve afectada por el desbalanceo de los datos
## - ¿Qué hay que cambiar en la función de ganancia para poder utilizarla? 
#Porque se ve afectada por el desbalanceo. Los baja+1 en realidad pesan mas que los baja+2 pq estan subrepresentados

## ---------------------------
## Step 6: Comparando tiempos con o sin muestras
## ---------------------------

t0 <- Sys.time()
r1 <- modelo_rpart(dtrain, dtest)
t1 <- Sys.time()
print("Train entero")
print(t1 - t0)
print(r1)

set.seed(semillas[1])
dtrain_sample <- tomar_muestra(dtrain)

t0 <- Sys.time()
r2 <- modelo_rpart(dtrain[dtrain_sample, ], dtest)
t1 <- Sys.time()
print("Muestra train")
print(t1 - t0)
print(r2)

## Preguntas
## - ¿Por qué sólo se muestrea train? pq no es costoso ESCORIAR sobre test

## ---------------------------
## Step 7: Buscando el mejor modelo con muestras aleatorias LHS
## ---------------------------

#Una función auxiliar para los experimentos

experimento_rpart <- function(ds, semillas, cp = 0, ms = 20, mb = 1, md = 10) {
  gan <- c() #antes: auc <- c()
  for (s in semillas) {
    set.seed(s)
    in_training <- caret::createDataPartition(ds$clase_binaria, p = 0.70,
        list = FALSE)
    train  <-  ds[in_training, ]
    test   <-  ds[-in_training, ]
    #train_sample <- tomar_muestra(train) sin muestra
    r <- modelo_rpart(train, test, #antes: train[train_sample,]
                    cp = cp, ms = ms, mb = mb, md = md)
    gan <- c(gan, r)
  }
  mean(gan)
}

# Haremos 25 experimentos aleatorios, armamos las muestras de acuerdo a como
# son las entradas de nuestro experimento.

set.seed(semillas[1])
cantidad_puntos <- 25
espacio_busqueda_1 <- optimumLHS(cantidad_puntos, 2)

# la primera columna es para el maxdepth, y la segunda para el minslip
espacio_busqueda_1[, 1] <- floor(15 * espacio_busqueda_1[, 1]) + 4
espacio_busqueda_1[, 2] <- floor(200 * espacio_busqueda_1[, 2]) + 2

resultados_random_search <- data.table()

for (e in 1:cantidad_puntos) {
  
  r <- experimento_rpart(dataset, semillas,
                        ms = espacio_busqueda_1[e, 2],
                        md = espacio_busqueda_1[e, 1])
  
  resultados_random_search <- rbindlist(list(resultados_random_search,
                  data.table(
                    md = espacio_busqueda_1[e, 1],
                    ms = espacio_busqueda_1[e, 2],
                    gan = r)
  ))
}

print(resultados_random_search)
ggplot(resultados_random_search, aes(x = md, y = ms, color = auc)) +
    scale_color_gradient(low = "blue", high = "red") +
    geom_point(aes(size = gan))


## Preguntas
## - ¿Hay alguna zona dónde parece que hay más ganancia?
## - ¿Cómo podemos continuar nuestra búsqueda?

###
### Break time =)
###

## ---------------------------
## Step 8: Trabajando con herramientas más profesionales
## ---------------------------
#Procesos gaussianos
#Construyen una funcion
#Toman puntos reales y van descubriendo nuevos

# Veamos un ejemplo
set.seed(semillas[1])
obj_fun <- makeSingleObjectiveFunction(
  name = "Sine",
  fn = function(x) sin(x),
  par.set = makeNumericParamSet(lower = 3, upper = 13, len = 1)
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 10L)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI(),
                           opt = "focussearch")

lrn <- makeMBOLearner(ctrl, obj_fun)
design <- generateDesign(6L, getParamSet(obj_fun), fun = lhs::maximinLHS)

run <- exampleRun(obj_fun, design = design, learner = lrn,
                 control = ctrl, points.per.dim = 100, show.info = TRUE)
#y seria la ganancia?
#El minimo lo buscaria en el espacio en el que puede llegar a estar el punto en mi proceso gausiano
#Los piquitos de abajo corresponden a los puntos rojos del grafico de arriba (partes grises mas bajas), nos indican donde explotar


# Ejecutar de a uno
plotExampleRun(run, iters = 1, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 2, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 3, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 5, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 6, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 7, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 8, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 9, densregion = TRUE, pause = FALSE)
plotExampleRun(run, iters = 10, densregion = TRUE, pause = FALSE)

## ---------------------------
## Step 9: Introduciendo la técnica en nuestro conjunto
## ---------------------------
#Existe una función que tenga sentido para optimizar?

resultados_maxdepth <- data.table()

for (v in 4:20) {
    r <- data.table(
      md = v,
      gan = experimento_rpart(dataset, semillas, md = v)
    )
    resultados_maxdepth <- rbindlist(list(resultados_maxdepth, r))
}

ggplot(resultados_maxdepth, aes(md, gan)) + geom_point()

## ---------------------------
## Step 10: Buscando con una Opt. Bayesiana para 1 parámetro
## ---------------------------

set.seed(semillas[1])
obj_fun_md <- function(x) {
  experimento_rpart(dataset, semillas, md = x$maxdepth) #se pasan parametros como listas
}

obj_fun <- makeSingleObjectiveFunction( #porque se pueden hacer funciones con mas de 1 metrica objetivo
  minimize = FALSE,
  fn = obj_fun_md,
  par.set = makeParamSet(
    makeIntegerParam("maxdepth",  lower = 4L, upper = 20L)
  ),
  # noisy = TRUE,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 10L)
ctrl <- setMBOControlInfill(
  ctrl,
  crit = makeMBOInfillCritEI(),
  opt = "focussearch",
  opt.focussearch.points = 2
)

#Por diseño busca 4 parametros?

lrn <- makeMBOLearner(ctrl, obj_fun)

surr_km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2")

run_md <- mbo(obj_fun, learner = surr_km, control = ctrl)
print(run_md)


## ---------------------------
## Step 11: Buscando con una Opt. Bayesiana para 2 parámetros
## ---------------------------

set.seed(semillas[1])
obj_fun_md_ms <- function(x) {
  experimento_rpart(dataset, semillas
            , md = x$maxdepth
            , ms = x$minsplit)
}

obj_fun <- makeSingleObjectiveFunction(
  minimize = FALSE,
  fn = obj_fun_md_ms,
  par.set = makeParamSet(
    makeIntegerParam("maxdepth",  lower = 4L, upper = 20L),
    makeIntegerParam("minsplit",  lower = 1L, upper = 200L)
    # makeNumericParam <- para parámetros continuos
  ),
  # noisy = TRUE,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 16L)
ctrl <- setMBOControlInfill(
  ctrl,
  crit = makeMBOInfillCritEI(),
  opt = "focussearch",
  # sacar parámetro opt.focussearch.points en próximas ejecuciones
  opt.focussearch.points = 20
)

lrn <- makeMBOLearner(ctrl, obj_fun)

surr_km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2")

run_md_ms <- mbo(obj_fun, learner = surr_km, control = ctrl, )
print(run_md_ms)

##### Si el mejor rdo lo encuentra en el tope del espacio de búsqueda, hay que agrandarlo!

## ---------------------------
## Step 12: Buscando con una Opt. Bayesiana para 3 parámetros
## ---------------------------
# cp = 0, ms = 20, mb = 1)


set.seed(semillas[1])
obj_fun_md_ms <- function(x) {
  experimento_rpart(dataset, semillas
                    , md = x$maxdepth
                    , ms = x$minsplit
                    , mb = x$minbucket
                    , cp = -1
  )
}

obj_fun <- makeSingleObjectiveFunction(
  minimize = FALSE,
  fn = obj_fun_md_ms,
  par.set = makeParamSet(
    makeIntegerParam("maxdepth",  lower = 3L, upper = 20L),
    makeNumericParam("minsplit" , lower=   1,   upper= 5000 ),
    makeNumericParam("minbucket", lower=   1,   upper= 1000 ),
    forbidden = quote( minbucket > 0.5*minsplit ) ## minbuket NO PUEDE ser mayor que la mitad de minsplit 
  ),
  # noisy = TRUE,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 100L)
ctrl <- setMBOControlInfill(
  ctrl,
  crit = makeMBOInfillCritEI(),
  #opt = "focussearch",
  # sacar parámetro opt.focussearch.points en próximas ejecuciones!!!!!
  #opt.focussearch.points = 20 #Podemos sacarlo
)

lrn <- makeMBOLearner(ctrl, obj_fun)

surr_km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2")

run_md_ms <- mbo(obj_fun, learner = surr_km, control = ctrl, )
print(run_md_ms)


## TAREA:
## Agregue todos los parámetros que considere. Una vez que tenga sus mejores
## parámetros, haga una copia del script rpart/z101_PrimerModelo.R, cambie los
## parámetros dentro del script, ejecutelo y suba a Kaggle su modelo.


#sacar train_sample tomar muestra train
#hay que entrenar con todos los datos
#Generar experimento rpart completo
#sin sample y cp -1
#Subir a kaggle
#poner ganacia en vez de auc

### RESULTADOS: K301_002
# Recommended parameters:
#   maxdepth=10; minsplit=1585.051; minbucket=479
# Objective: y = 19058666.667
# 
# Optimization path
# 12 + 100 entries in total, displaying last 10 (or less):
#   maxdepth minsplit minbucket        y dob eol error.message exec.time         ei error.model train.time prop.type propose.time        se     mean
# 103        9 1792.677  474.9161 18669333  91  NA          <NA>     24.58 -22701.667        <NA>       0.17 infill_ei         1.53  93597.51 18851123
# 104        7 1373.290  474.4385 18725333  92  NA          <NA>     21.15 -30836.806        <NA>       0.14 infill_ei         1.64 346800.22 18550463
# 105       10 1512.059  473.7001 18884000  93  NA          <NA>     26.19 -21207.128        <NA>       0.37 infill_ei         1.33 179665.34 18739747
# 106       10 1517.543  491.6839 18752000  94  NA          <NA>     26.42 -30380.067        <NA>       0.15 infill_ei         1.38 238055.45 18703024
# 107        9 2322.267  478.2723 18885333  95  NA          <NA>     24.30 -20924.490        <NA>       0.41 infill_ei         1.45  35750.89 18897121
# 108        8 1517.773  474.5723 18884000  96  NA          <NA>     23.53 -22513.481        <NA>       0.45 infill_ei         1.47  68815.18 18874813
# 109        8 1490.469  453.1400 18440000  97  NA          <NA>     30.78 -15934.290        <NA>       0.21 infill_ei         1.54 273180.20 18562718
# 110       10 1556.279  477.0433 19052000  98  NA          <NA>     31.29 -19558.869        <NA>       0.27 infill_ei         1.48  46380.55 18887408
# 111       10 1585.051  478.9056 19058667  99  NA          <NA>     33.51 -20333.001        <NA>       0.27 infill_ei         1.53  26836.73 19067693
# 112       10 1029.927  480.0643 17234667 100  NA          <NA>     34.72  -6592.537        <NA>       0.36 infill_ei         1.66 439996.12 18275157