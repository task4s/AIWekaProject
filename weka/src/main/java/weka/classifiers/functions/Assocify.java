/*
 * Logic for classification algorithm,
 * based on associative rules.
 */

package weka.classifiers.functions;

import weka.classifiers.Classifier;
import weka.core.FastVector;
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
import weka.core.*;
import weka.associations.*;

public class Assocify extends Classifier
                        implements WeightedInstancesHandler, OptionHandler {


    /** number of possible classes */
    private int m_ClassCount;
    
    /** class Attribute */
    private Attribute m_Class;
    
    /** contains the distributions of the classes in the learning set */
    private double [] m_ClassDistribution;

    /** contains the instances in the learning set */
    private Instances m_LearningSet;

    /** contains the association rules mined */
    private FastVector [] m_AssociationRules;

    /** contains each association class coverage used to evaluate   */
    /** instance class                                              */
    private double[][] m_ClassScores;

    /** number of rules mined by the association algorithm */
    private int m_numRules;

    /** min metric, required to mine the rules */
    private double m_minMetric;

    /** are there any association rules mined */
    private Boolean m_RulesMined;

    /** association rules miner */
    private Apriori m_Apriori;

    /** init options */
    public Assocify()
    {
        m_numRules = 70;
        m_minMetric = 0.9;
        m_RulesMined = false;
    }

     /**
     * Returns a string describing this classifier
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo()
    {
        return "Algorithm, using assciation rules to predict instance class";
    }

    public Enumeration listOptions()
    {
        String string1 = "\tThe number of rules to be mined by the association algorithm (default: " + m_numRules + ")",
                string2 = "\tThe min metric, to be satisfied (default: " + m_minMetric + ")";

        FastVector optionsVector = new FastVector(2);

        optionsVector.addElement(new Option(string1, "N", 1, "-N <required number of rules to be mined>"));
        optionsVector.addElement(new Option(string2, "C", 1, "-C <minimum metric score for a rule>"));

        return optionsVector.elements();
    }

    public String[] getOptions()
    {
        String[] options = new String[4];
        int current = 0;
        options[current++] = "-N"; options[current++] = "" + m_numRules;
        options[current++] = "-C"; options[current++] = "" + m_minMetric;

        while (current < options.length)
        {
            options[current++] = "";
        }
        return options;
    }

    public void setOptions(String[] options) throws Exception
    {
        String a_numRules = Utils.getOption('N',options),
                a_minMetric = Utils.getOption('C',options);

        if(a_numRules.length() != 0)
            m_numRules = Integer.parseInt(a_numRules);
        if(a_minMetric.length() != 0)
            m_minMetric = (new Double(a_minMetric)).doubleValue();
    }

    public String toString()
    {
        String result = "Association rules classifier initiates classifier\n\n";
        result += "Association rules classifier tries to mine association rules\n\n";
        if(m_RulesMined)
        {
            result += m_Apriori.toString();
        }
        else
        {
            result += "No Rules were mined with minMetric: " + m_minMetric;
        }

        return result;
    }
    
    public Capabilities getCapabilities()
    {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        //attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);        
        result.enable(Capability.MISSING_VALUES);
        
        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        return result;
    }
    
    public void buildClassifier(Instances instances) throws Exception
    {
        getCapabilities().testWithFail(instances);
        
        m_LearningSet = instances;
        m_ClassCount = m_LearningSet.numClasses();
        m_Class = m_LearningSet.classAttribute();
        
        EvaluateClassDistribution();
        
        m_Apriori = new Apriori();
        m_Apriori.setNumRules(this.m_numRules);
        m_Apriori.setMinMetric(this.m_minMetric);
        m_Apriori.buildAssociations(m_LearningSet);

        m_AssociationRules = m_Apriori.getAllTheRules();
        FastVector premises = m_AssociationRules[0];
        if(premises.size() > 0)
        {
            m_RulesMined = true;
        }

        EvaluateRulesClassDistribution();
    }
    


    

     public double classifyInstance(Instance instance) throws Exception
     {
         double [] instanceScores = new double [m_ClassCount];
         for(int i = 0; i < m_ClassCount; i++)
         {
             instanceScores[i] = m_ClassDistribution[i];
         }

         FastVector premises = m_AssociationRules[0];
         FastVector consequences = m_AssociationRules[1];

         for(int rule = 0; rule < m_numRules; rule ++)
         {
             AprioriItemSet premise = (AprioriItemSet)premises.elementAt(rule);
             AprioriItemSet consequence = (AprioriItemSet)consequences.elementAt(rule);
             if(premise.containedBy(instance) && consequence.containedBy(instance))
             {
                 for(int i = 0; i < m_ClassCount; i++)
                     instanceScores[i] *= m_ClassScores[rule][i];
             }
         }

         double returnClass = 0;
         double currentScore = 0;
         for(int i = 0; i < m_ClassCount; i++)
         {
             if(currentScore < instanceScores[i])
             {
                 returnClass = (double)i;
                 currentScore = instanceScores[i];
             }
         }

         return returnClass;
     }

     private void EvaluateRulesClassDistribution()
     {
        m_ClassScores = new double[m_numRules][];

        FastVector premises = m_AssociationRules[0];
        FastVector consequences = m_AssociationRules[1];
        m_numRules = premises.size();
        for(int i = 0; i< m_numRules; i++)
        {
            double [] numerators = new double[m_ClassCount];
            double [] denominators = new double[m_ClassCount];
            for(int pos = 0; pos < m_ClassCount; pos++)
            {
                numerators[pos] = denominators[pos] = 0.0;
            }

            Enumeration enu = m_LearningSet.enumerateInstances();
            while (enu.hasMoreElements())
            {
                Instance instance = (Instance) enu.nextElement();
                if (!instance.classIsMissing())
                {
                    int classIndex = (int)instance.classValue();
                    denominators[classIndex] += 1;
                    AprioriItemSet premise = (AprioriItemSet)premises.elementAt(i);
                    AprioriItemSet consequence = (AprioriItemSet)consequences.elementAt(i);
                    if(premise.containedBy(instance) && consequence.containedBy(instance))
                        numerators[classIndex] += 1;
                }
            }

            double [] classScores = new double [m_ClassCount];
            for(int pos = 0; pos < m_ClassCount; pos++)
            {
                classScores[pos] = numerators[pos] / denominators[pos];
            }

            m_ClassScores[i] = classScores;
        }
     }

     private void EvaluateClassDistribution()
     {
        m_ClassDistribution = new double[m_ClassCount];
        for(int i =0; i < m_ClassCount; i++)
            m_ClassDistribution[i] = 0;

        Enumeration enu = m_LearningSet.enumerateInstances();
        double sumOfWeights = 0.0;

        while (enu.hasMoreElements())
        {
            Instance instance = (Instance) enu.nextElement();
            if (!instance.classIsMissing())
            {

                    m_ClassDistribution[(int)instance.classValue()] += instance.weight();
                    sumOfWeights += instance.weight();
            }
        }

        for(int i =0; i < m_ClassCount; i++)
            m_ClassDistribution[i] /= sumOfWeights;
    }

     public static void main(String argv[]) {
        runClassifier(new Assocify(), argv);
    }
}
