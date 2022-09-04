#--- 1. setup----

#cargo las librerias que necesito
require("data.table")
require("rpart")
require("ROCR")
require("ggplot2")
require("lubridate")
require("lhs")
require("DiceKriging")
require("mlrMBO")
require("rgenoud")

#Aqui se debe poner la carpeta de la materia de SU computadora local
#setwd("D:\\gdrive\\UBA2022\\")  #Establezco el Working Directory
setwd("C:/Users/PC/Documents/DMEyF/")

semillas  <- c(100043, 100049, 100153, 100169,100183)

#cargo el dataset
dataset  <- fread("./datasets/competencia1_2022.csv")

dtrain  <- dataset[ foto_mes==202101 ]  #defino donde voy a entrenar

dapply  <- dataset[ foto_mes==202103 ]  #defino donde voy a aplicar el modelo

# Creamos una clase binaria
dataset[, clase_binaria := ifelse(
  clase_ternaria == "BAJA+2",
  "evento",
  "noevento"
)]
# Borramos el target viejo
dataset[, clase_ternaria := NULL]

#Partición estratificada
in_training <- caret::createDataPartition(dataset$clase_binaria,
                                          p = 0.70, list = FALSE)
dtrain  <-  dataset[in_training, ]
dtest   <-  dataset[-in_training, ]

#--- 2. arbol promedio----

## Step 2: Nuestra pizza: Un modelo


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

#--- 3. estimacion tiempo de busqueda----

## Step 3: There Ain't No Such Thing As A Free Lunch


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

#--- 4. exploración de tipos de sampleo----

## Step 4: Empezando a probar con menos casos

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

#--- 5. sampleo y eleccion metricas----

## Step 5: Tomando una muestra de sangre

# Armamos una función para modelar con el fin de simplificar el código futuro
modelo_rpart <- function(train, test, cp =  0, ms = 20, mb = 1, md = 10) {
  modelo <- rpart(clase_binaria ~ ., data = train,
                  xval = 0,
                  cp = cp,
                  minsplit = ms,
                  minbucket = mb,
                  maxdepth = md)
  
  test_prediccion <- predict(modelo, test, type = "prob")
  roc_pred <-  ROCR::prediction(test_prediccion[, "evento"],
                                test$clase_binaria,
                                label.ordering = c("noevento", "evento"))
  auc_t <-  ROCR::performance(roc_pred, "auc")
  
  unlist(auc_t@y.values)
}

# Armamos una función para modelar con el fin de simplificar el código futuro
modelo_rpart_ganancia <- function(train, test, cp =  0, ms = 20, mb = 1, md = 10) {
  modelo <- rpart(clase_binaria ~ ., data = train,
                  xval = 0,
                  cp = cp,
                  minsplit = ms,
                  minbucket = mb,
                  maxdepth = md)
  
  test_prediccion <- predict(modelo, test, type = "prob")
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
## - ¿Por qué se eligió usar el AUC? no se ve afectada por el desbalanceo de los datos.
## - ¿Qué hay que cambiar en la función de ganancia para poder utilizarla?

#--- 6. comparación entrenamiento c/sin muestreo----

## Step 6: Comparando tiempos con o sin muestras

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
## - ¿Por qué sólo se muestrea train?


#--- 7. modelando con hyperp LHS ----

## Step 7: Buscando el mejor modelo con muestras aleatorias LHS

# Una función auxiliar para los experimentos
experimento_rpart <- function(ds, semillas, cp = 0, ms = 20, mb = 1, md = 10) {
  auc <- c()
  for (s in semillas) {
    set.seed(s)
    in_training <- caret::createDataPartition(ds$clase_binaria, p = 0.70,
                                              list = FALSE)
    train  <-  ds[in_training, ]
    test   <-  ds[-in_training, ]
    train_sample <- tomar_muestra(train)
    r <- modelo_rpart(train[train_sample,], test, 
                      cp = cp, ms = ms, mb = mb, md = md)
    auc <- c(auc, r)
  }
  mean(auc)
}

# Una función auxiliar para los experimentos
experimento_rpart_completo <- function(ds, semillas, cp = -1, ms = 20, mb = 1, md = 10) {
  gan <- c()
  for (s in semillas) {
    set.seed(s)
    in_training <- caret::createDataPartition(ds$clase_binaria, p = 0.70,
                                              list = FALSE)
    train  <-  ds[in_training, ]
    test   <-  ds[-in_training, ]
    #train_sample <- tomar_muestra(train)
    r <- modelo_rpart_ganancia(train, test, 
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
                                               auc = r)
  ))
}

print(resultados_random_search)
ggplot(resultados_random_search, aes(x = md, y = ms, color = auc)) +
  scale_color_gradient(low = "blue", high = "red") +
  geom_point(aes(size = auc))


## Preguntas
## - ¿Hay alguna zona dónde parece que hay más ganancia?
## - ¿Cómo podemos continuar nuestra búsqueda?

#--- 8. testeo dummy optimización bayesiana----

## Step 8: Trabajando con herramientas más profesionales - modelos bayesianos


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


#--- 9. testeo optimización bayesiana 1 param----


## Introduciendo la técnica en nuestro conjunto

resultados_maxdepth <- data.table()

for (v in 4:20) {
  r <- data.table(
    md = v,
    auc = experimento_rpart(dataset, semillas, md = v)
  )
  resultados_maxdepth <- rbindlist(list(resultados_maxdepth, r))
}

ggplot(resultados_maxdepth, aes(md, auc)) + geom_point()

## Buscando con una Opt. Bayesiana para 1 parámetro

set.seed(semillas[1])
obj_fun_md <- function(x) {
  experimento_rpart(dataset, semillas, md = x$maxdepth)
}

obj_fun <- makeSingleObjectiveFunction(
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

lrn <- makeMBOLearner(ctrl, obj_fun)

surr_km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2")

run_md <- mbo(obj_fun, learner = surr_km, control = ctrl)
print(run_md)

#--- 10. testeo optimización bayesiana 2 param----


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

#--- 11. testeo optimización bayesiana 3 param----

## Step 11: Mas opt. Bayesiana para 3 parámetros


set.seed(semillas[1])
obj_fun_md_ms_mb <- function(x) {
  experimento_rpart_completo(dataset, semillas
                             , md = x$maxdepth
                             , ms = x$minsplit
                             , mb = floor(x$minsplit*x$minbucket))
}

obj_fun <- makeSingleObjectiveFunction(
  minimize = FALSE,
  fn = obj_fun_md_ms_mb,
  par.set = makeParamSet(
    makeIntegerParam("maxdepth",  lower = 2L, upper = 30L),
    makeIntegerParam("minsplit",  lower = 1L, upper = 2000L),
    makeNumericParam("minbucket",  lower = 0L, upper = 1L)
    #makeNumericParam("cp",  lower = -1, upper = 1L)
    # makeNumericParam <- para parámetros continuos
  ),
  noisy = TRUE,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl()
ctrl <- setMBOControlTermination(ctrl, iters = 150L) #sugerencia Gustavo
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

print("Final paso 11")

# Recommended parameters:
#   maxdepth=5; minsplit=1048; minbucket=0.65
# Objective: y = 19213333.333

# Optimization path
# 12 + 150 entries in total, displaying last 10 (or less):
#   maxdepth minsplit   minbucket        y dob eol error.message exec.time         ei error.model train.time prop.type propose.time
# 153       29     1189 0.203803068 17654667 141  NA          <NA>     50.42 -17538.182        <NA>       0.22 infill_ei         0.14
# 154        5     1440 0.559356593 17432000 142  NA          <NA>     31.20  -3902.106        <NA>       0.33 infill_ei         0.13
# 155       10      869 0.579695588 15597333 143  NA          <NA>     42.06  -6463.196        <NA>       0.43 infill_ei         0.12
# 156       25     1944 0.881908345 17525333 144  NA          <NA>     41.19 -34673.705        <NA>       0.44 infill_ei         0.14
# 157       14      580 0.774246368 16866667 145  NA          <NA>     46.57  -3137.041        <NA>       0.46 infill_ei         0.13
# 158        8     1589 0.438608581 17662667 146  NA          <NA>     37.60  -2394.350        <NA>       0.49 infill_ei         0.22
# 159        7     1215 0.001212321 16876000 147  NA          <NA>     39.81  -1972.264        <NA>       0.34 infill_ei         0.14
# 160       19     1507 0.521143638 16670667 148  NA          <NA>     44.06  -1523.700        <NA>       0.50 infill_ei         0.13
# 161       30     1837 0.760316262 17785333 149  NA          <NA>     42.28 -52596.661        <NA>       0.34 infill_ei         0.13
# 162       28     1783 0.622792251 17301333 150  NA          <NA>     42.65  -5651.775        <NA>       0.44 infill_ei         0.12
# se     mean
# 153  650097.0 18214718
# 154  669527.3 17782203
# 155  611282.8 18041452
# 156 1297017.2 17215871
# 157  772724.5 17463438
# 158  762755.6 17419197
# 159 1182693.0 16186998
# 160  695956.0 17493377
# 161 1235008.5 17569263
# 162  817609.8 17516088

#genero el archivo para Kaggle
#primero creo la carpeta donde va el experimento
dir.create( "./exp/" )
dir.create( "./exp/KA2001" )

fwrite( dapply[ , list(numero_de_cliente, Predicted) ], #solo los campos para Kaggle
        file= "./exp/KA2001/K101_001.csv",
        sep=  "," )
