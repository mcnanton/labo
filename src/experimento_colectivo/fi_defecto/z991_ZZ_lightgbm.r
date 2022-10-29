#Necesita para correr en Google Cloud
# 128 GB de memoria RAM
# 256 GB de espacio en el disco local
#   8 vCPU


#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")

require("lightgbm")

#Parametros del script
PARAM  <- list()
PARAM$experimento  <- "ZZ9410_ec_fi"
PARAM$exp_input  <- "HT9410_ec_fi"
PARAM$corte = 9000

PARAM$modelos  <- 1 # No necesito los top n modelos, solo 1
# FIN Parametros del script

#ksemilla  <- 102191

vector_semillas <- c(100043, 100049, 100153, 100169, 100183, 100184, 100044, 100050, 100154, 100170)

#------------------------------------------------------------------------------
options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa

base_dir <- "~/buckets/b1/"

#creo la carpeta donde va el experimento
dir.create( paste0( base_dir, "exp/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd(paste0( base_dir, "exp/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO

#leo la salida de la optimizaciob bayesiana
arch_log  <- paste0( base_dir, "exp/", PARAM$exp_input, "/BO_log.txt" )
tb_log  <- fread( arch_log )
setorder( tb_log, -ganancia )

#leo el nombre del expermento de la Training Strategy
arch_TS  <- paste0( base_dir, "exp/", PARAM$exp_input, "/TrainingStrategy.txt" )
TS  <- readLines( arch_TS, warn=FALSE )

#leo el dataset donde voy a entrenar el modelo final
arch_dataset  <- paste0( base_dir, "exp/", TS, "/dataset_train_final.csv.gz" )
dataset  <- fread( arch_dataset )

#leo el dataset donde voy a aplicar el modelo final
arch_future  <- paste0( base_dir, "exp/", TS, "/dataset_future.csv.gz" )
dfuture <- fread( arch_future )


#defino la clase binaria
dataset[ , clase01 := ifelse( clase_ternaria %in% c("BAJA+1","BAJA+2"), 1, 0 )  ]

## Entiendo que no hace falta que pise el contenido de clase_ternaria porque esta linea se ocupa de omitir las clases
campos_buenos  <- setdiff( colnames(dataset), c( "clase_ternaria", "clase01") )

# Gracias a Lucas Trevisani por la inspiración en su repo sobre cómo armar esto:

for (i in vector_semillas)
  {
  ganancia_promedio_modelo <- c()

  #genero un modelo para cada semilla
  for( i in  1:PARAM$modelos )
  {
    parametros  <- as.list( copy( tb_log[ i ] ) ) #parametros de la BO
    iteracion_bayesiana  <- parametros$iteracion_bayesiana
  
    arch_modelo  <- paste0( "modelo_" ,
                            sprintf( "%02d", i ),
                            "_",
                            sprintf( "%03d", iteracion_bayesiana ),
                            ".model" )
  
  
    #creo CADA VEZ el dataset de lightgbm
    dtrain  <- lgb.Dataset( data=    data.matrix( dataset[ , campos_buenos, with=FALSE] ),
                            label=   dataset[ , clase01],
                            weight=  dataset[ , ifelse( clase_ternaria %in% c("BAJA+2"), 1.0000001, 1.0)],
                            free_raw_data= FALSE
                          )
  
    ganancia  <- parametros$ganancia
  
    #elimino los parametros que no son de lightgbm
    parametros$experimento  <- NULL
    parametros$cols         <- NULL
    parametros$rows         <- NULL
    parametros$fecha        <- NULL
    parametros$prob_corte   <- NULL
    parametros$estimulos    <- NULL
    parametros$ganancia     <- NULL
    parametros$iteracion_bayesiana  <- NULL
  
    #Utilizo la semilla DE MI VECTOR DE 10 SEMILLAS que estoy loopeando
    parametros$seed  <- i
    
    #genero el modelo entrenando en los datos finales
    set.seed( parametros$seed )
    
    modelo_final  <- lightgbm( data= dtrain,
                               param=  parametros,
                               verbose= -100 )
  
    # #grabo el modelo, achivo .model
    # lgb.save( modelo_final,
    #           file= arch_modelo )
  
    #creo y grabo la importancia de variables
    # tb_importancia  <- as.data.table( lgb.importance( modelo_final ) )
    # fwrite( tb_importancia,
    #         file= paste0( "impo_", 
    #                       sprintf( "%02d", i ),
    #                       "_",
    #                       sprintf( "%03d", iteracion_bayesiana ),
    #                       ".txt" ),
    #         sep= "\t" )
  
  
    #genero la prediccion, Scoring
    prediccion  <- predict( modelo_final,
                            data.matrix( dfuture[ , campos_buenos, with=FALSE ] ) )
  
    tb_prediccion  <- dfuture[  , list( numero_de_cliente, foto_mes ) ]
    #tb_prediccion[ , prob := prediccion ]
  
    tbl  <- dfuture[ , list(clase_ternaria) ]
    
    tbl[ , prob := prediccion ]
    
    setorder( tbl, -prob )
    
    tbl[  , Predicted := 0L ]
    tbl[ 1:PARAM$corte, Predicted := 1L ]
    
    ganancia_test  <- tbl[ prob >= prob_corte, sum( ifelse(clase_ternaria=="BAJA+2", 78000, -2000 ) )]
    ganancia_iter <- rbind( ganancia_iter, ganancia_test)
    # nom_pred  <- paste0( "pred_",
    #                      sprintf( "%02d", i ),
    #                      "_",
    #                      sprintf( "%03d", iteracion_bayesiana),
    #                      ".csv"  )
    # 
    # fwrite( tb_prediccion,
    #         file= nom_pred,
    #         sep= "\t" )
  
 
    # nom_submit  <- paste0( PARAM$experimento, 
    #                        "_",
    #                        sprintf( "%02d", i ),
    #                        "_",
    #                        sprintf( "%03d", iteracion_bayesiana ),
    #                        "_",
    #                        sprintf( "%05d", corte ),
    #                        ".csv" )
    # fwrite(  tb_prediccion[ , list( numero_de_cliente, Predicted ) ],
    #            file= nom_submit,
    #            sep= "," )
  
    #borro y limpio la memoria para la vuelta siguiente del for
    rm( tb_prediccion )
    rm( tb_importancia )
    rm( modelo_final)
    rm( parametros )
    #rm( dtrain )
    gc()
    
    # Evaluo la predicción de la semilla i
    ganancia_promedio_semilla <- mean(ganancia_iter)
    
    # Guardo la ganancia de la semilla i junto a las ganancias de las otras semillas
    ganancia_promedio_modelo <- rbind(ganancia_promedio_modelo, ganancia_promedio_semilla)
    
  }
}

# Analizo la ganancia promedio de el modelo

