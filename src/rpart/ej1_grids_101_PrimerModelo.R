#Arbol elemental con libreria  rpart
#Debe tener instaladas las librerias  data.table  ,  rpart  y  rpart.plot

#cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")

#Aqui se debe poner la carpeta de la materia de SU computadora local
#setwd("D:\\gdrive\\UBA2022\\")  #Establezco el Working Directory
setwd("C:/Users/PC/Documents/DMEyF/")
semillas <- c(100043, 100049, 100153, 100169, 100183)


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


# Armamos una función que nos calcule la ganancia, usando el punto de corte de
# 0.025
ganancia <- function(probabilidades, clase) {
  return(sum(
    (probabilidades >= 0.025) * ifelse(clase == "evento", 78000, -2000))
  )
}

# inicializo tabla de rdos de grid search
resultados_grid_search <- data.table()

# Complete los valores que se van a combinar para cada parámetro a explorar

for (cp in c(-1, 0.01, 0.005, -0.1)) {
  for (md in c(3, 5, 10, 15, 20, 30)) {
    for (ms in c(1, 30, 50, 80, 100, 120)) {
      for (mb in c(1, as.integer(ms / 2))) {
        
        t0 <- Sys.time()
        gan_semillas <- c()
        for (s in semillas) {
          set.seed(s)
          in_training <- caret::createDataPartition(dataset[, get("clase_binaria")],
                                                    p = 0.70, list = FALSE)
          dtrain  <-  dataset[in_training, ]
          dtest   <-  dataset[-in_training, ]
          
          modelo <- rpart(clase_binaria ~ .,  #quiero predecir binaria a partir de el resto de las variables
                          data = dtrain,
                          xval = 0,
                          cp = cp,
                          minsplit = ms,
                          minbucket = mb,
                          maxdepth = md)
          
          pred_testing <- predict(modelo, dtest, type = "prob")
          gan <- ganancia(pred_testing[, "evento"], dtest$clase_binaria) / 0.3
          
          gan_semillas <- c(gan_semillas, gan)
        }
        tiempo <-  as.numeric(Sys.time() - t0, units = "secs")
        
        resultados_grid_search <- rbindlist(list(
          resultados_grid_search,
          data.table(
            tiempo = tiempo,
            cp = cp,
            mb = mb,
            ms = ms,
            md = md,
            gan = mean(gan_semillas))
        ))
      }
    }
  }
}

# Visualizo los parámetros de los mejores parámetros
View(resultados_grid_search[gan == max(gan), ])

#genero el modelo,  aqui se construye el arbol
 modelo  <- rpart(formula=   clase_binaria ~ .,  #quiero predecir clase_ternaria a partir de el resto de las variables
                  data=      dtrain,  #los datos donde voy a entrenar
                  xval=      0,
                  cp=       -0.1,   #esto significa no limitar la complejidad de los splits
                  minsplit=  80,     #minima cantidad de registros para que se haga el split
                  minbucket= 40,     #tamaño minimo de una hoja
                  maxdepth=  5 )    #profundidad maxima del arbol


#grafico el arbol
#prp(modelo, extra=101, digits=5, branch=1, type=4, varlen=0, faclen=0)


#aplico el modelo a los datos nuevos
prediccion  <- predict( object= modelo,
                        newdata= dapply,
                        type = "prob")

#prediccion es una matriz con TRES columnas, llamadas "BAJA+1", "BAJA+2"  y "CONTINUA"
#cada columna es el vector de probabilidades 

#agrego a dapply una columna nueva que es la probabilidad de BAJA+2
dapply[ , prob_baja2 := prediccion[, "evento"] ]

#solo le envio estimulo a los registros con probabilidad de BAJA+2 mayor  a  1/40
dapply[ , Predicted := as.numeric( prob_baja2 > 1/40 ) ]

#genero el archivo para Kaggle
#primero creo la carpeta donde va el experimento
#dir.create( "./exp/" )
#dir.create( "./exp/KA2001" )

fwrite( dapply[ , list(numero_de_cliente, Predicted) ], #solo los campos para Kaggle
        file= "./exp/KA2001/K101_vp.csv",
        sep=  "," )
