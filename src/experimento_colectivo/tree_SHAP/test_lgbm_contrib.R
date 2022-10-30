#limpio la memoria
rm( list=ls() )
gc()

require("data.table")
require("Matrix")
require("lightgbm")

#cargo el dataset
dataset <- fread( "C:/Users/PC/Documents/DMEyF/datasets/competencia3_2022.csv.gz")

dataset  <- dataset[ foto_mes %in% c(202103, 202105) ]
gc()

#agrego a mis fieles canaritos
# nada temo porque Ellos son mis centinelas y delataran a los embusteros
for( i in 1:20 )  dataset[ , paste0( "canarito", i ) := runif(nrow(dataset)) ]

dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]

campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01", "foto_mes" ) )

azar  <- runif( nrow(dataset) )
dataset[ , entrenamiento := foto_mes>= 202101 &  foto_mes<= 202103  & ( clase01==1 | azar < 0.10 ) ]

dtrain  <- lgb.Dataset( data=    data.matrix(  dataset[ entrenamiento==TRUE, campos_buenos, with=FALSE]),
                        label=   dataset[ entrenamiento==TRUE, clase01],
                        weight=  dataset[ entrenamiento==TRUE, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)],
                        free_raw_data= FALSE
)

dvalid  <- lgb.Dataset( data=    data.matrix(  dataset[ foto_mes==202105, campos_buenos, with=FALSE]),
                        label=   dataset[ foto_mes==202105, clase01],
                        weight=  dataset[ foto_mes==202105, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)],
                        free_raw_data= FALSE
)


param <- list( objective= "binary",
               metric= "custom",
               first_metric_only= TRUE,
               boost_from_average= TRUE,
               feature_pre_filter= FALSE,
               verbosity= -100,
               seed= 999983,
               max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
               min_gain_to_split= 0.0, #por ahora, lo dejo fijo
               lambda_l1= 0.0,         #por ahora, lo dejo fijo
               lambda_l2= 0.0,         #por ahora, lo dejo fijo
               max_bin= 31,            #por ahora, lo dejo fijo
               num_iterations= 9999,   #un numero muy grande, lo limita early_stopping_rounds
               force_row_wise= TRUE,    #para que los alumnos no se atemoricen con tantos warning
               learning_rate= 0.065, 
               feature_fraction= 1.0,   #lo seteo en 1 para que las primeras variables del dataset no se vean opacadas
               min_data_in_leaf= 260,
               num_leaves= 60,
               early_stopping_rounds= 200 )

modelo  <- lgb.train( data= dtrain,
                      valids= list( valid= dvalid ),
                      eval= fganancia_lgbm_meseta,
                      param= param,
                      verbose= -100 )