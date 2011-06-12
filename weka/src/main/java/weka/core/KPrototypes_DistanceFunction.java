/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.core;
import java.util.Vector;

/**
 *
 * @author Todor Tsonkov
 */
public class KPrototypes_DistanceFunction extends NormalizableDistance
  implements Cloneable  {

  private double gamma;

    /** for serialization. */
  private static final long serialVersionUID = 1068606253458807903L;

  public KPrototypes_DistanceFunction( double gamma){
        this.gamma = gamma;
  }

  public KPrototypes_DistanceFunction() {
    super();
    this.gamma = 0;
  }

  public KPrototypes_DistanceFunction(Instances data, double gamma) {
    super(data);
    this.gamma = gamma;
  }

  /**
   * Returns a string describing this object.
   *
   * @return 		a description of the evaluator suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return
        "Implementing KPrototypes distance (or similarity) function.\n\n"
      + "One object defines not one distance but the data model in which "
      + "the distances between objects of that data model can be computed.\n\n"
      + "Attention: For efficiency reasons the use of consistency checks "
      + "(like are the data models of the two instances exactly the same), "
      + "is low.\n\n";
  }

    //MOST IMPORTANT PART!
  /**
   * Calculates the distance between two instances.
   *
   * @param first 	the first instance
   * @param second 	the second instance
   * @return 		the distance between the two given instances
   */
  @Override
  public double distance(Instance first, Instance second) {
     double sum_nominal = 0.0;
     double sum_continuous = 0.0;
     
      for(int i = 0; i < first.numAttributes(); i++){
        if(first.attribute(i).isNominal()){
            if(first.attribute(i).equals(second.attribute(i)) )
                sum_nominal+=1.0;
        }else
        if (first.attribute(i).isNumeric()){
            sum_continuous += distance(first, second, Double.POSITIVE_INFINITY);
        }
      }
     
      return sum_continuous + gamma*sum_nominal;
  }
public String[] getOptions() {
    Vector<String>	result;
    
    result = new Vector<String>();

    if (getDontNormalize())
      result.add("-D");
    
    result.add("-R");
    result.add(getAttributeIndices());
    
    if (getInvertSelection())
      result.add("-V");

    return result.toArray(new String[result.size()]);
  }


  //TODO: IMPLEMENT?
  protected double updateDistance(double currDist, double diff) {
    double	result;

    result  = currDist;
    result += diff * diff;

    return result;
  }

  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.13 $");
  }
    
}
