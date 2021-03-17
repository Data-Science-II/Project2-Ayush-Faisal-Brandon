package neuralnet

import scalation.analytics.{TranRegression, Perceptron, NeuralNet_3L, NeuralNet_XL}
import scalation.analytics.PredictorMat
import scalation.analytics.HyperParameter
import scalation.analytics.Optimizer
import scalation.analytics.ActivationFun._
import scalation.analytics.Fit._
import scalation.columnar_db.Relation
import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD} 
import scalation.plot.{PlotM, Plot}
import scalation.util.banner

object Auto extends App {

    val autoMPG = Relation.apply("data/auto-mpg.csv", "autoMPG", 
                                domain=null, key = 0, eSep = ",", cPos = null)
    autoMPG.show(5)

    val (x,y) = autoMPG.toMatriDD(1 to 6, 0)
    val ox = new MatrixD(x.dim1, 1, 1.0) ++^ x // x augmented with vector of ones
    val normX = new MatrixD(ox.normalizeU()) //normalizing features 
    val normY: VectoD = VectorD(y.normalize()) //normalizing response
    
    /*
    //Perceptron Model
    val perceptronHP = new HyperParameter()
    perceptronHP += ("eta", 0.05, 0.1) //optimal eta 
    perceptronHP += ("bSize", 100, 20)
    perceptronHP += ("maxEpochs", 10000, 100)

    val perceptron = new Perceptron(normX, normY, hparam = perceptronHP, f0 = f_reLU)

    banner("Perceptron")
    perceptron.train().eval()
    println(perceptron.summary)

    //3LayerNetwork Model
    banner("3Layer NN")
    

    val nn3LHparam = new HyperParameter()
    nn3LHparam += ("eta", 0.05, 0.1)
    nn3LHparam += ("bSize", 20, 20)  
    nn3LHparam += ("lambda", 0, 0)
    nn3LHparam += ("maxEpochs", 1000, 1000)
    
    val nn3L = new NeuralNet_3L(normX, matNormY, 10, hparam = nn3LHparam, f0 = f_tanh, f1 = f_id)
    var trained_model = nn3L.train0()
    for (i <- 0 to 20) { 
        trained_model = trained_model.train0()
    }  //for i 
    println(trained_model.eval().report)
    */
    val matNormY: MatrixD = new MatrixD(new MatrixD(autoMPG.toMatriD(Seq(0))).normalizeU()) //NormY as matrix
    //4LayerNetwork Model
    banner("4Layer NN")
    val nn4L = new NeuralNet_XL(normX, matNormY, Array(8, 4))
    nn4L.reset(0.0005)
    var trained_4model = nn4L.train0()
    for (i <- 0 to 50) { 
        trained_4model = trained_4model.train0()
    }  //for i 

    println(trained_4model.eval().report)



} //App