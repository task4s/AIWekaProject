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
   * replace missing values in training instances
   */
  private ReplaceMissingValues m_ReplaceMissingFilter;

  /**
   * number of clusters to generate
   */
  private int m_NumClusters = 2;

  /**
   * Holds the standard deviations of the numeric attributes in each cluster
   */
  private Instances m_ClusterStdDevs;

  /**
   * For each cluster, holds the frequency counts for the values of each
   * nominal attribute
   */
  private int [][][] m_ClusterNominalCounts;
  private int[][] m_ClusterMissingCounts;

  /**
   * Stats on the full data set for comparison purposes
   * In case the attribute is numeric the value is the mean if is
   * being used the Euclidian distance or the median if Manhattan distance
   * and if the attribute is nominal then it's mode is saved
   */
  private double[] m_FullMeansOrMediansOrModes;
  private double[] m_FullStdDevs;
  private int[][] m_FullNominalCounts;
  private int[] m_FullMissingCounts;

  /**
   * Display standard deviations for numeric atts
   */
  private boolean m_displayStdDevs;

  /**
   * Replace missing values globally?
   */
  private boolean m_dontReplaceMissing = false;

  /**
   * The number of instances in each cluster
   */
  private int [] m_ClusterSizes;

  /**
   * Maximum number of iterations to be executed by the K-means subalgorithm
   */
  private int m_MaxIterations = 500;

  /**
   * Holds the squared errors for all clusters
   */
  private double [] m_squaredErrors;

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

  /**
   * Number of executions of the K-means subalgorithm at each splitting
   */
  private int m_NumExecutions = 2;

  /**
   * the default constructor
   */
  public BisectingKMeans() {
    super();

    m_SeedDefault = 10;
    setSeed(m_SeedDefault);
  }

  /**
   * Returns a string describing this clusterer
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Cluster data using the k means algorithm. Can use either "
      + "the Euclidean distance (default) or the Manhattan distance."
      + " If the Manhattan distance is used, then centroids are computed "
      + "as the component-wise median rather than mean.";
  }

  /**
   * Returns default capabilities of the clusterer.
   *
   * @return      the capabilities of this clusterer
   */
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

  public void buildClusterer(Instances data) throws Exception {
    // TODO: implement algorithm here?
    getCapabilities().testWithFail(data);

    //what is this?
    //m_ReplaceMissingFilter = new ReplaceMissingValues();
    Instances instances = new Instances(data);
  }

  public int numberOfIterations() throws Exception {
    return m_MaxIterations;
  }

  public void setNumIterations(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Number of clusters must be > 0");
    }
    m_MaxIterations = n;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numClustersTipText() {
    return "set number of clusters";
  }

  /**
   * set the number of clusters to generate
   *
   * @param n the number of clusters to generate
   * @throws Exception if number of clusters is negative
   */
  public void setNumClusters(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Number of clusters must be > 0");
    }
    m_NumClusters = n;
  }

  /**
   * gets the number of clusters to generate
   *
   * @return the number of clusters to generate
   */
  public int getNumClusters() {
    return m_NumClusters;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String maxIterationsTipText() {
    return "set maximum number of iterations";
  }

  /**
   * set the maximum number of iterations to be executed
   *
   * @param n the maximum number of iterations
   * @throws Exception if maximum number of iteration is smaller than 1
   */
  public void setMaxIterations(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Maximum number of iterations must be > 0");
    }
    m_MaxIterations = n;
  }

  /**
   * gets the number of maximum iterations to be executed
   *
   * @return the number of clusters to generate
   */
  public int getMaxIterations() {
    return m_MaxIterations;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numExecutionsTipText() {
    return "set number of executions of the K-means subalgorithm";
  }

  /**
   * set the maximum number of iterations to be executed
   *
   * @param n the maximum number of iterations
   * @throws Exception if maximum number of iteration is smaller than 1
   */
  public void setNumExecutions(int n) throws Exception {
    if (n <= 0) {
      throw new Exception("Number of executions must be > 0");
    }
    m_MaxIterations = n;
  }

  /**
   * gets the number of maximum iterations to be executed
   *
   * @return the number of clusters to generate
   */
  public int getNumExecutions() {
    return m_NumExecutions;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String displayStdDevsTipText() {
    return "Display std deviations of numeric attributes "
      + "and counts of nominal attributes.";
  }

  /**
   * Sets whether standard deviations and nominal count
   * Should be displayed in the clustering output
   *
   * @param stdD true if std. devs and counts should be
   * displayed
   */
  public void setDisplayStdDevs(boolean stdD) {
    m_displayStdDevs = stdD;
  }

  /**
   * Gets whether standard deviations and nominal count
   * Should be displayed in the clustering output
   *
   * @return true if std. devs and counts should be
   * displayed
   */
  public boolean getDisplayStdDevs() {
    return m_displayStdDevs;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String dontReplaceMissingValuesTipText() {
    return "Replace missing values globally with mean/mode.";
  }

  /**
   * Sets whether missing values are to be replaced
   *
   * @param r true if missing values are to be
   * replaced
   */
  public void setDontReplaceMissingValues(boolean r) {
    m_dontReplaceMissing = r;
  }

  /**
   * Gets whether missing values are to be replaced
   *
   * @return true if missing values are to be
   * replaced
   */
  public boolean getDontReplaceMissingValues() {
    return m_dontReplaceMissing;
  }

  /**
   * Returns the tip text for this property.
   *
   * @return 		tip text for this property suitable for
   *         		displaying in the explorer/experimenter gui
   */
  public String distanceFunctionTipText() {
    return "The distance function to use for instances comparison " +
      "(default: weka.core.EuclideanDistance). ";
  }

  /**
   * returns the distance function currently in use.
   *
   * @return the distance function
   */
  public DistanceFunction getDistanceFunction() {
    return m_DistanceFunction;
  }

  /**
   * sets the distance function to use for instance comparison.
   *
   * @param df the new distance function to use
   * @throws Exception if instances cannot be processed
   */
  public void setDistanceFunction(DistanceFunction df) throws Exception {
    if(!(df instanceof EuclideanDistance) &&
       !(df instanceof ManhattanDistance))
      {
        throw new Exception("BisectingKMeans currently only supports the Euclidean and Manhattan distances.");
      }
    m_DistanceFunction = df;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String preserveInstancesOrderTipText() {
    return "Preserve order of instances.";
  }

  /**
   * Sets whether order of instances must be preserved
   *
   * @param r true if missing values are to be
   * replaced
   */
  public void setPreserveInstancesOrder(boolean r) {
    m_PreserveOrder = r;
  }

  /**
   * Gets whether order of instances must be preserved
   *
   * @return true if missing values are to be
   * replaced
   */
  public boolean getPreserveInstancesOrder() {
    return m_PreserveOrder;
  }

  /**
   * Returns the number of clusters.
   *
   * @return the number of clusters generated for a training dataset.
   * @throws Exception if number of clusters could not be returned
   * successfully
   */
  public int numberOfClusters() throws Exception {
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
  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   *
   * <pre> -N &lt;num&gt;
   *  number of clusters.
   *  (default 2).
   * </pre>
   *
   * <pre> -V
   *  Display std. deviations for centroids.
   * </pre>
   *
   * <pre> -M
   *  Replace missing values with mean/mode.
   * </pre>
   *
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 10)
   * </pre>
   *
   * <pre> -A &lt;classname and options&gt;
   *  Distance function to be used for instance comparison
   *  (default weka.core.EuclidianDistance)
   * </pre>
   *
   * <pre> -I &lt;num&gt;
   *  Maximum number of iterations of the K-means subalgorithm.
   * </pre>
   *
   * <pre> -O
   *  Preserve order of instances.
   * </pre>
   *
   * <pre> -X
   *  Number of executions of the K-means subalgorithm at each splitting.
   * </pre>
   *
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
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

    optionString = Utils.getOption("X", options);
    if (optionString.length() != 0) {
      setNumExecutions(Integer.parseInt(optionString));
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

  /**
   * Gets the current settings of BisectingKMeans
   *
   * @return an array of strings suitable for passing to setOptions()
   */
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

    result.add("-X");
    result.add(""+ getNumExecutions());

    if(m_PreserveOrder){
      result.add("-O");
    }

    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Gets the standard deviations of the numeric attributes in each cluster
   *
   * @return		the standard deviations of the numeric attributes
   * 			in each cluster
   */
  public Instances getClusterStandardDevs() {
    return m_ClusterStdDevs;
  }

  /**
   * Returns for each cluster the frequency counts for the values of each
   * nominal attribute
   *
   * @return		the counts
   */
  public int [][][] getClusterNominalCounts() {
    return m_ClusterNominalCounts;
  }

  /**
   * Gets the squared error for all clusters
   *
   * @return		the squared error
   */
  public double getSquaredError() {
    return Utils.sum(m_squaredErrors);
  }

  /**
   * Gets the number of instances in each cluster
   *
   * @return		The number of instances in each cluster
   */
  public int [] getClusterSizes() {
    return m_ClusterSizes;
  }

  /**
   * Gets the assignments for each instance
   * @return Array of indexes of the centroid assigned to each instance
   * @throws Exception if order of instances wasn't preserved or no assignments were made
   */
  public int [] getAssignments() throws Exception{
    if(!m_PreserveOrder){
      throw new Exception("The assignments are only available when order of instances is preserved (-O)");
    }
    if(m_Assignments == null){
      throw new Exception("No assignments made.");
    }
    return m_Assignments;
  }

  /**
   * Returns the revision string.
   *
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1000 $");
  }

  public static void main (String[] argv) {
    runClusterer(new BisectingKMeans(), argv);
  }

}
