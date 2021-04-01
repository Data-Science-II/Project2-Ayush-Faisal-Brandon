package neuralnet

import scalation.analytics.{TranRegression, Perceptron, NeuralNet_3L, NeuralNet_XL}
import scalation.analytics.PredictorMat
import scalation.analytics.HyperParameter
import scalation.analytics.Optimizer
import scalation.analytics.ActivationFun._
import scalation.analytics.Fit._
import scalation.math._
import scalation.columnar_db.Relation
import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD} 
import scalation.plot.{PlotM, Plot}
import scalation.util.banner

object SeoulBike extends App {

    val bike = Relation.apply("data/SeoulBikeDataCleaned.csv", "bike", 
                                domain=null, key = 0, eSep = ",", cPos = null)
    bike.show(5)

    val (x,y) = bike.toMatriDD(1 to 14, 0)
    val ox = new MatrixD(x.dim1, 1, 1.0) ++^ x // x augmented with vector of ones
    val normX = new MatrixD(ox.normalizeU()) //normalizing features 
    val normY: VectoD = VectorD(y.normalize()) //normalizing response
    val matY: MatrixD = new MatrixD(bike.toMatriD(Seq(0))) //NormY as matrix
    val matNormY: MatrixD = new MatrixD(matY.normalizeU())

    val trg = new TranRegression(ox, y)
    banner("Transformed Regression")
    println(trg.analyze().summary)
    var (_, trgForwards) = trg.forwardSelAll()
    var (_, trgBackwards) = trg.backwardElimAll()
    //var (_, trgStep) = trg.stepRegressionAll()

    plot_and_save(trgForwards, "ForwardTranReg.png")
    plot_and_save(trgBackwards, "BackwardTranReg.png")
    //plot_and_save(trgStep, "StepTranReg.png")
    
    //Perceptron Model
    val perceptronHP = new HyperParameter()
    perceptronHP += ("eta", 0.000000005, 0.1) //optimal eta 
    perceptronHP += ("bSize", 300, 10)
    perceptronHP += ("maxEpochs", 10000, 100)

    val perceptron = new Perceptron(normX, normY, hparam = perceptronHP, f0 = f_reLU)

    banner("Perceptron")
    perceptron.train().eval()
    println(perceptron.summary)

    var (_, perceptronForwards) = perceptron.forwardSelAll()
    var (_, perceptronBackwards) = perceptron.backwardElimAll()
    //var (_, perceptronStep) = perceptron.stepRegressionAll()

    plot_and_save(perceptronForwards, "ForwardPerceptron.png")
    plot_and_save(perceptronBackwards, "BackwardPerceptron.png")
    //plot_and_save(perceptronStep, "StepPerceptron.png")


    //3LayerNetwork Model
    
    banner("3Layer NN")
    
    
    val nn3LHparam = new HyperParameter()
    nn3LHparam += ("eta", 0.075, 0.1)
    nn3LHparam += ("bSize", 20, 20)  
    nn3LHparam += ("lambda", 0, 0)
    nn3LHparam += ("maxEpochs", 1000, 1000)

    val nn3L = new NeuralNet_3L(normX, matNormY, 4, hparam = nn3LHparam, f0 = f_tanh, f1 = f_tanh)
    var trained_model = nn3L.train()
    println(trained_model.eval().report)

    var (_, forwards) = nn3L.forwardSelAll()
    var (_, backwards) = nn3L.backwardElimAll()
    //var (_, step) = nn3L.stepRegressionAll()

    plot_and_save(forwards, "Forward3L.png")
    plot_and_save(backwards, "Backward3L.png")
    //plot_and_save(step, "Step3L.png")
    
    
    
    
    //4LayerNetwork Model
    banner("4Layer NN")
    val nn4L = new NeuralNet_XL(normX, matNormY, Array(8, 4))
    nn4L.reset(0.0005)
    var trained_4model = nn4L.train()


    println(trained_4model.eval().report)
    var (_, forwards4) = nn4L.forwardSelAll()
    var (_, backwards4) = nn4L.backwardElimAll()
    //var (_, step4) = nn4L.stepRegressionAll()

    plot_and_save(forwards4, "Forward4L.png")
    plot_and_save(backwards4, "Backward4L.png")
    //plot_and_save(step4, "Step4L.png")    


    /**
    * Plots and saves the resutls of the backwards/forwards 
    * variable selection process. 
    *
    * @param regMat the matrix that results from the process 
    * @param path the path to be saved at 
    */
    def plot_and_save (regMat: MatriD, path: String, basePath: String = "plots/scala/SeoulBike/")  = { 
        val plot = new PlotM(VectorD.range(0, regMat.dim1), regMat.t, 
            label = Array[String]("R^2", "adj-R^2", "cvR^2"), 
            _title = "Quality of Fit vs. Model Complexity", 
            lines = true)
        
        plot.saveImage(basePath + path) 

    } //plot_and_save

} //App