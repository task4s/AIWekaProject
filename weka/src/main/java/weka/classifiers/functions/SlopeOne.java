/*
 * SlopeOne regression algorithm
 *
 */
package weka.classifiers.functions;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;


import java.util.*;
import java.io.*;

public class SlopeOne extends Classifier implements WeightedInstancesHandler, OptionHandler {

    /** Array for storing coefficients of linear regression. */
    private double[] m_Coefficients;
    /** Array for storing the average coefficients of linear regression. */
    private double[] m_CoefficientsAverage;
    /**Number of examples for each couple of attributes */
    private double[] m_Frequency;
    /** Which attributes are relevant? */
    private boolean[] m_SelectedAttributes;
    /** Variable for storing transformed training data. */
    private Instances m_TransformedData;
    /** The filter storing the transformation from nominal to
    binary attributes. */
    private NominalToBinary m_TransformFilter;
    /** The index of the class attribute */
    private int m_ClassIndex;
    /** checks true of we are dealing woth movielens-like data*/
    private boolean movieLensData = false;
    /**The data stored for easy user-based searches*/
    HashMap<Double, HashMap<Double, Double>> dataMovies;
   
    /**
     * Returns a string describing this classifier
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Slope one regresion algorithm";
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(4);
        newVector.addElement(new Option("\tSpecifies whether movielens dataset is present.\n"
                + "\t(default no)",
                "X", 0, "-X"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options. <p/>
     *
    <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -D
     *  Produce debugging output.
     *  (default no debugging output)</pre>
     *
     * <pre> -S &lt;number of selection method&gt;
     *  Set the attribute selection method to use. 1 = None, 2 = Greedy.
     *  (default 0 = M5' method)</pre>
     *
     * <pre> -C
     *  Do not try to eliminate colinear attributes.
     * </pre>
     *
     * <pre> -R &lt;double&gt;
     *  Set ridge parameter (default 1.0e-8).
     * </pre>
     *
    <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String selectionString = Utils.getOption('X', options);
        if (selectionString.length() != 0) {
            movieLensData = true;
        } else {
            movieLensData = false;
        }

    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {

        String[] options = new String[1];
        int current = 0;

        if (movieLensData) {
            options[current++] = "-X";
            options[current++] = "" + movieLensData;
        } else {
            options[current++] = "-X false";
        }

        return options;
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Builds a SlopeOne regression model given the supplied training data.
     *
     * @param insts the training data.
     * @throws Exception if an error occurs
     */
    public void buildClassifier(Instances data) throws Exception {

        if (movieLensData == false) {
            getCapabilities().testWithFail(data);

            // remove instances with missing class
            data = new Instances(data);
            data.deleteWithMissingClass();

            // Preprocess instances
            m_TransformFilter = new NominalToBinary();
            m_TransformFilter.setInputFormat(data);
            data = Filter.useFilter(data, m_TransformFilter);

            m_TransformedData = data;
            m_ClassIndex = data.classIndex();
            
            // Turn all attributes on or off
            m_SelectedAttributes = new boolean[data.numAttributes()];
            int j=0;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (i != m_ClassIndex /*&& j<newData.numAttributes()&& data.attribute(i).equals(newData.attribute(j))*/) {
                    m_SelectedAttributes[i] = true;
                    ++j;
                }
                else
                    m_SelectedAttributes[i] = false;
            }




            m_Coefficients = null;

            m_Frequency = new double[data.numAttributes()];
            m_Coefficients = new double[data.numAttributes()];
            m_CoefficientsAverage = new double[data.numAttributes()];

            Enumeration enu = data.enumerateInstances();

            Instance instance;

            while (enu.hasMoreElements()) {
                instance = (Instance) enu.nextElement();
                //if(!instance.isMissing(m_ClassIndex))
                for (int i = 0; i < instance.numAttributes(); ++i) {
                    
                    if (!instance.isMissing(i) && i != m_ClassIndex&& m_SelectedAttributes[i]) {
                        m_Frequency[i] += instance.weight();
                        m_Coefficients[i] += (instance.value(m_ClassIndex) - instance.value(i)) * instance.weight();
                    }
                }
            }
            for (int i = 0; i < data.numAttributes(); ++i) {
                if(m_Frequency[i]!=0)
                    m_CoefficientsAverage[i] = m_Coefficients[i] / m_Frequency[i];
            }
        } else {
            m_TransformedData = data;
            buildClassifierMovieLens(m_TransformedData);
        }
        // Save memory
        //m_TransformedData = new Instances(data, 0);
    }

    private void buildClassifierMovieLens(Instances data) {
        dataMovies = new HashMap<Double,HashMap<Double, Double>>();
        //dataMoviesMov = new HashMap<Double,HashMap<Double, Double>>();
        Enumeration enu = data.enumerateInstances();

        Instance instance;

        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            for (int i = 0; i < 3; ++i) {
                if (!dataMovies.containsKey(instance.value(0))) {
                    dataMovies.put(instance.value(0),
                            new HashMap<Double, Double>());
                }
                //TODO dataMoviesMov
                dataMovies.get(instance.value(0)).put(instance.value(1), instance.value(2));

            }
        }
    }

    public double classifyInstance(Instance instance) throws Exception {

        if (movieLensData) {
            return regressionPredictionMovieLens(instance);
        }
        // Transform the input instance
        Instance transformedInstance = instance;

        m_TransformFilter.input(transformedInstance);
        m_TransformFilter.batchFinished();
        transformedInstance = m_TransformFilter.output();

        // Calculate the dependent variable from the regression model
        return regressionPrediction(transformedInstance);
    }

    private double regressionPrediction(Instance transformedInstance)
            throws Exception {
        double result = 0;
        int column = 0;
        for (int j = 0; j < transformedInstance.numAttributes(); j++) {
            if ((m_ClassIndex != j)
                    && (m_SelectedAttributes[j]) && !transformedInstance.isMissing(j)) {
                result = result + (m_CoefficientsAverage[j] + transformedInstance.value(j)) * m_Frequency[j];
                column += m_Frequency[j];
            }
        }
        result /= column;
        return result;
    }

    private double regressionPredictionMovieLens(Instance input) {
        HashMap<Double, Double> ratingsMade = new HashMap<Double,Double>(dataMovies.get(input.value(0)));
        //HashMap<Double, Double> usersRating = new HashMap<Double,Double>(dataMovies.get(input.value(0)));

     /*   if (ratingsMade.containsKey(input.value(1))) {
            return ratingsMade.get(input.value(1));
        } */
        
        double result = 0;
        double goalMovie = input.value(1);
        double myUser = input.value(0);

        LinkedList<Double> closeUsers=new LinkedList<Double>();

        Enumeration enumer = m_TransformedData.enumerateInstances();
        Instance instance;

        /*Mean difference and Freaquency in movies-ratings data*/
        HashMap<Double, Double> diffArr=new HashMap<Double, Double>();
        HashMap<Double, Double> diffArrNum=new HashMap<Double, Double>();


        while (enumer.hasMoreElements()) {
            instance = (Instance) enumer.nextElement();
            for (int i = 0; i < 3; ++i) {
                if (ratingsMade.containsKey(instance.value(1)) 
                        && instance.value(1) != goalMovie && instance.value(0)!=myUser) {
                    if (!closeUsers.contains(instance.value(0))
                            && dataMovies.get(instance.value(0)).containsKey(goalMovie)) {
                        closeUsers.add(instance.value(0));
                    }
                }
            }
        }
        HashMap<Double, Double> ratings;
        while (!closeUsers.isEmpty()) {
            ratings = dataMovies.get(closeUsers.pop());
            for (Double count : ratings.keySet()) {
                if (ratingsMade.containsKey(count) && count != goalMovie) {
                    if (diffArr.containsKey(count)) {
                        double tmp = diffArr.remove(count);
                        diffArr.put(count,tmp + ratings.get(count) - ratings.get(goalMovie));

                        tmp = diffArrNum.remove(count);
                        diffArrNum.put(count, tmp + 1);
                    }
                    else
                    {
                        diffArr.put(count, ratings.get(count) - ratings.get(goalMovie));
                        diffArrNum.put(count, 1.0);
                    }
                }
            }
        }
        int count=0;
        for (Double cnt : ratingsMade.keySet()) {
            if(diffArr.containsKey(cnt))
            {
                result += ratingsMade.get(cnt) - diffArr.get(cnt) / diffArrNum.get(cnt);
                ++count;
            }
        }
        return result/count;
    }

    public String toString() {

        if (m_TransformedData == null) {
            return "SlopeOne: No model built yet.";
        }
        try {
            StringBuffer text = new StringBuffer();

            text.append("\nSlopeOne Regression Model\n\n");
            if(!movieLensData)
            {
            boolean first = true;
            int num_Att = 0;
            text.append(m_TransformedData.classAttribute().name() + " =     (\n\n");
            for (int i = 0; i < m_TransformedData.numAttributes(); i++) {
                if ((i != m_ClassIndex)
                        && (m_SelectedAttributes[i])) {
                    if (!first) {
                        text.append(" +\n\n");
                    } else {
                        first = false;
                    }
                    text.append(m_TransformedData.attribute(i).name() + " ");
                    if (m_CoefficientsAverage[i] > 0) {
                        text.append("+");
                    }
                    text.append(Utils.doubleToString(m_CoefficientsAverage[i], 4));
                    ++num_Att;
                }
            }
            text.append(" ) /" + num_Att + ";\n");
            }
            else
                text.append("MovieLens Data Detected.");
            return text.toString();
        } catch (Exception e) {
            return "Can't print SlopeOne Regression!";
        }
    }

    public static void main(String argv[]) {
        runClassifier(new SlopeOne(), argv);
    }
}
