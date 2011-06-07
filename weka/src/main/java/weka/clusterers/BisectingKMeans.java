/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.clusterers;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Random;
import java.util.Vector;

/**
 *
 * @author MetalGearRex
 */
public class BisectingKMeans
  extends RandomizableClusterer 
  implements NumberOfClustersRequestable, WeightedInstancesHandler {

  /** for serialization */
  static final long serialVersionUID = -3235809600124455376L;

  /**
   * number of clusters to generate
   */
  private int m_NumClusters = 2;

  /**
   * holds the cluster centroids
   */

  /**
   * Holds the standard deviations of the numeric attributes in each cluster
   */

  /**
   * For each cluster, holds the frequency counts for the values of each
   * nominal attribute
   */

  /**
   * Stats on the full data set for comparison purposes
   * In case the attribute is numeric the value is the mean if is
   * being used the Euclidian distance or the median if Manhattan distance
   * and if the attribute is nominal then it's mode is saved
   */
  
  /**
   * Display standard deviations for numeric atts
   */
  private boolean m_displayStdDevs;

  /**
   * Replace missing values globally?
   */
  private boolean m_dontReplaceMissing = false;

  /**
   * Maximum number of iterations to be executed
   */
  private int m_MaxIterations = 500;

  /** the distance function used. */
  protected DistanceFunction m_DistanceFunction = new EuclideanDistance();

  /**
   * Preserve order of instances
   */
  private boolean m_PreserveOrder = false;

  /**
   * Assignments obtained
   */
  protected int[] m_Assignments = null;


  private int m_NumIterations = 2;

  public int numberOfIterations() throws Exception {
    return m_NumIterations;
  }
    public void setNumIterations(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Number of clusters must be > 0");
    }
    m_NumIterations = n;
  }

    public int numberOfClusters() throws Exception {
    return m_NumClusters;
    }
    public void setNumClusters(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Number of clusters must be > 0");
    }
    m_NumClusters = n;
  }

  public void buildClusterer(Instances data) throws Exception {
      // TODO: implement algorithm here?
    }

    public void setMaxIterations(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Maximum number of iterations must be > 0");
    }
    m_MaxIterations = n;
  }

  public int getNumClusters() {
    return m_NumClusters;
  }

   public Enumeration listOptions () {
    Vector result = new Vector();

    result.addElement(new Option(
                                 "\tTsonkov Cholakov.\n"
                                 + "\t(default 2).",
                                 "N", 1, "-N <num>"));
    result.addElement(new Option(
                                 "\tDisplay std. deviations for centroids.\n",
                                 "V", 0, "-V"));
    result.addElement(new Option(
                                 "\tReplace missing values with mean/mode.\n",
                                 "M", 0, "-M"));

    result.add(new Option(
                          "\tDistance function to use.\n"
                          + "\t(default: weka.core.EuclideanDistance)",
                          "A", 1,"-A <classname and options>"));

    result.add(new Option(
                          "\tMaximum number of iterations.\n",
                          "I",1,"-I <num>"));

    result.addElement(new Option(
                                 "\tPreserve order of instances.\n",
                                 "O", 0, "-O"));

    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());

    return  result.elements();
  }

  public void setDistanceFunction(DistanceFunction df) throws Exception {
    if(!(df instanceof EuclideanDistance) && 
       !(df instanceof ManhattanDistance))
      {
        throw new Exception("SimpleKMeans currently only supports the Euclidean and Manhattan distances.");
      }
    m_DistanceFunction = df;
  }
  
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
    result.enable(Capability.NO_CLASS);

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    return result;
  }

  public int getMaxIterations() {
    return m_MaxIterations;
  }

   public void setOptions (String[] options)
    throws Exception {

    m_displayStdDevs = Utils.getFlag("V", options);
    m_dontReplaceMissing = Utils.getFlag("M", options);

    String optionString = Utils.getOption('N', options);

    if (optionString.length() != 0) {
      setNumClusters(Integer.parseInt(optionString));
    }

    optionString = Utils.getOption("I", options);
    if (optionString.length() != 0) {
      setMaxIterations(Integer.parseInt(optionString));
    }

    String distFunctionClass = Utils.getOption('A', options);
    if(distFunctionClass.length() != 0) {
      String distFunctionClassSpec[] = Utils.splitOptions(distFunctionClass);
      if(distFunctionClassSpec.length == 0) {
        throw new Exception("Invalid DistanceFunction specification string.");
      }
      String className = distFunctionClassSpec[0];
      distFunctionClassSpec[0] = "";

      setDistanceFunction( (DistanceFunction)
                           Utils.forName( DistanceFunction.class,
                                          className, distFunctionClassSpec) );
    }
    else {
      setDistanceFunction(new EuclideanDistance());
    }

    m_PreserveOrder = Utils.getFlag("O", options);

    super.setOptions(options);
  }

   public String[] getOptions () {
    int       	i;
    Vector    	result;
    String[]  	options;

    result = new Vector();

    if (m_displayStdDevs) {
      result.add("-V");
    }

    if (m_dontReplaceMissing) {
      result.add("-M");
    }

    result.add("-N");
    result.add("" + getNumClusters());

    result.add("-A");
    result.add((m_DistanceFunction.getClass().getName() + " " +
                Utils.joinOptions(m_DistanceFunction.getOptions())).trim());

    result.add("-I");
    result.add(""+ getMaxIterations());

    if(m_PreserveOrder){
      result.add("-O");
    }

    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    return (String[]) result.toArray(new String[result.size()]);
  }

}
