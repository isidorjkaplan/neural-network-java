package me.izzyk.neuralnet;


import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.text.NumberFormat;
import java.util.*;

/**
 * The neural network class IS-A Vector in N dimensions depending on the amount of neurons.
 */
public class NeuralNet extends Vector {

    /**
     * The constructor to read a Neural Network from a file
     * @param file The file to read it from
     * @throws IOException
     */
    public NeuralNet(File file) throws IOException {
        this(new Scanner(file));
    }

    /**
     * A constructor for reading a seralized network from a string
     * @param saved The seralzed network
     * @throws IOException
     */
    public NeuralNet(String saved) throws IOException {
        this(new Scanner(saved));
    }

    /**
     * The constructor to read a network from a scanner
     * @param sc The scanner to read it from
     * @throws IOException
     */
    public NeuralNet(Scanner sc) throws IOException {
        super(getLayersFromFile(sc));
        for (int i = 1; i < this.getWeightings().length; i++) {
            for (int j = 0; j < this.getWeightings()[i].length; j++) {
                getBiases()[i][j] = sc.nextDouble();
                for (int k = 0; k < this.getWeightings()[i - 1].length; k++) {
                    getWeightings()[i][j][k] = sc.nextDouble();
                }
            }
        }
        sc.close();
    }

    /**
     * The constructor to create a neural network based on the amount of layers
     * @param layers The amount of layers
     */
    public NeuralNet(int... layers) {
        super(layers);
        for (int i = 0; i < this.getWeightings().length; i++) {
            for (int j = 0; j < this.getWeightings()[i].length; j++) {
                for (int k = 0; k < this.getWeightings()[i][j].length; k++) {
                    this.getWeightings()[i][j][k] = 2 * Math.random() - 1;
                }
            }
        }
    }



    /**
     * A constructor to get the layer count from a scanner, namely the first 4 integers that show up.
     * @param sc The scanner
     * @return The first 4 integers
     */
    private static int[] getLayersFromFile(Scanner sc) {
        int[] layers = new int[sc.nextInt()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = sc.nextInt();
        }
        return layers;
    }

    /**
     * The Sigmoid Squishifaction function (non linear compression function) that compresses the entire real number line to a number between 0 and 1
     * @param value The value to compress
     * @return The compressed function
     */
    protected static double sigmoid(double value) {
        return 1 / (1 + Math.pow(Math.E, -value));
    }

    /**
     * The derivitave of the sigmoid function, gives the "instantious slope" at a value
     * @param value The value to get the derivitave at
     * @return The derivitave of sigmoid(value)
     */
    protected static double sigmoid_prime(double value) {
        double e = Math.pow(Math.E, -value);
        return e / Math.pow(1 + e, 2);
    }

    /**
     * The inverse function of the sigmoid function
     * @param value The value
     * @return The inverse function
     */
    protected static double inverse_sigmoid(double value) {
        return -Math.log(1 / value - 1);
    }

    /**
     * The amount of input neurons on the network
     * @return The amount of input neurons on the network
     */
    public int getInputsCount() {
        return getBiases()[0].length;
    }

    /**
     * A funtion for the amount of output neurons on the network
     * @return The amount of output neurons on the network
     */
    public int getOutputsCount() {
        return getBiases()[getBiases().length-1].length;
    }

    /**
     * The total amount of neural pathways
     * @return The total amount of neural pathways
     */

    public int getPathwayCount() {
        int pathways = 0;
        for (int i = getWeightings().length - 1; i > 0; i--) {
            pathways += getWeightings()[i].length * getWeightings()[i - 1].length;
        }
        return pathways;
    }

    /**
     * A function to dynamically remove a neuron from the network
     * @param layer The layer to add it
     * @param id The ID
     */
    protected void removeNeuron(int layer, int id) {
        //TODO
    }

    /**
     * A function to dynmaiclly add a neuron to the network
     * @param layer The layer to add the neuron to
     */
    protected void addNeuron(int layer) {
        if (layer <= 0 || layer >= getBiases().length - 1) {
            throw new InputMismatchException("You must enter a neuron in the hidden layers (1 to output-1): " + layer);
        }
        int id = getBiases()[layer].length;
        getBiases()[layer] = Arrays.copyOf(getBiases()[layer], id + 1);
        getBiases()[layer][id] = 0;
        getWeightings()[layer] = Arrays.copyOf(getWeightings()[layer], id + 1);
        getWeightings()[layer][id] = new double[getWeightings()[layer][id - 1].length];
        for (int i = 0; i < getWeightings()[layer][id].length; i++) {
            getWeightings()[layer][id][i] = Math.random() * 2 - 1;
        }
        for (int id2 = 0; id2 < getWeightings()[layer + 1].length; id2++) {
            getWeightings()[layer + 1][id2] = Arrays.copyOf(getWeightings()[layer + 1][id2], id + 1);
            getWeightings()[layer + 1][id2][id] = Math.random() * 2 - 1;
        }
    }

    /**
     * A function to save the network to a string
     * @return The network saved to a string
     */
    public String saveToString() {
        List<String> file = new ArrayList<>();
        file.add(getWeightings().length + "");
        String header = getWeightings()[0].length + "";
        for (int i = 1; i < getWeightings().length; i++) {
            header += " " + getWeightings()[i].length;
            //file.add("Layer: " + i);
            for (int j = 0; j < getBiases()[i].length; j++) {
                String line = getBiases()[i][j] + "";// + j;
                for (int k = 0; k < getWeightings()[i][j].length; k++) {
                    line += " " + getWeightings()[i][j][k];
                }
                file.add(line);
            }
        }
        file.add(1, header);
        String string = "";
        for (String s: file) {
            string += s + "\n";
        }
        return string;
    }

    /**
     * A function to save the network to a file
     * @param path The file to save it to
     * @throws IOException
     */
    public void save(File path) throws IOException {
        path.createNewFile();
        FileOutputStream out = new FileOutputStream(path);
        out.write(saveToString().getBytes());
        out.close();
    }

    /**
     * A function to get an active instance of the network given a set of inputs
     * @param values The inputs
     * @return The active network object
     */
    public ActiveNetwork getInstance(double[] values) {
        return this.new ActiveNetwork(values);
    }

    /**
     * The ActiveNetwork class represents an active network
     */
    public class ActiveNetwork {
        /**
         * The array of values for a given [layer][id] pair
         */
        public double[][] values;

        /**
         * The constructor to create an active network from a set of inputs
         * @param inputs The set of inputs
         */
        public ActiveNetwork(double[] inputs) {
            values = new double[getBiases().length][];
            for (int layer = 0; layer < values.length; layer++) {
                values[layer] = new double[getBiases()[layer].length];
                for (int id = 0; id < values[layer].length; id++) {
                    values[layer][id] = -1;
                }
            }
            //System.out.println(values[0].length);
            for (int i = 0; i < values[0].length; i++) {
                //System.out.println(i + " " + values[0].length + " " + inputs.length);
                values[0][i] =
                        inputs[i];
            }
            for (int i = 0; i < values[values.length - 1].length; i++) {
                getValue(values.length - 1, i);
            }
        }

        /**
         * The function to get the value of a given neuron
         * @param layer The layer
         * @param id The neuron id
         * @return The value of that neuron
         */

        public double getValue(int layer, int id) {
            if (values[layer][id] == -1) {
                double value = 0;
                for (int target = 0; target < values[layer - 1].length; target++) {
                    value += getWeightings()[layer][id][target] * getValue(layer - 1, target);
                }
                value += getBiases()[layer][id];
                value = sigmoid(value);
                //value+=neuron.getBias();
                values[layer][id] = value;
            }
            return values[layer][id];
        }

        /**
         * A function to get the values of the inputs
         * @return The values of the inputs
         */
        public double[] getInputs() {
            return values[0];
        }

        /**
         * A fnction to get the value of the outputs
         * @return The value of the outputs
         */
        public double[] getOutputs() {
            return values[values.length - 1];
        }

    }

    /**
     * A class that represents a peice of test data for the network to learn
     */
    public abstract class TestData {
        /**
         * The correct input values
         */
        private double[] inputs;
        /**
         * The correct output values
         */
        private double[] outputs;

        /**
         * The constructo rto create a peice of test data
         * @param inputs The inputs
         * @param outputs The outputs
         */
        public TestData(double[] inputs, double[] outputs) {
            this.inputs = inputs;
            this.outputs = outputs;
        }

        /**
         * The function to check if a given output is "correct"
         * @param actual The given output
         * @return If it is correct
         */
        public abstract boolean isCorrect(double[] actual);

        /**
         * The inputs
         * @return The inputs
         */
        public double[] getInputs() {
            return inputs;
        }

        /**
         * The correct outputs
         * @return The correct outputs
         */
        public double[] getOutputs() {
            return outputs;
        }
    }


    /**
     * The class that teaches a network newpeices of data using the gradiant decent learning algorythem
     */
    public class NetworkTeacher {
        /**
         * The prefix for displaying the obtained data
         */
        private String outputPrefix = "%-SIZEs %-SIZEs %-SIZEs %-SIZEs %-SIZEs %-SIZEs %-SIZEs".replace("SIZE", "10");
        /**
         * The test dat for the network to learn
         */
        private Set<TestData> testData;
        /**
         * The shuffled test data
         */
        private List<Set<TestData>> shuffled;
        /**
         * The amount of steps taken since hte creation of the object
         */
        private int steps = 0;
        /**
         * The learning factor, DO NOT PLAY WITH UNLESS YOU KNOW WHAT YOU ARE DOING
         */
        private double learningFactor = 1;
        /**
         * The momentum factor, ALSO DO NOT PLAY WITH
         */
        private double momentumFactor = 0.9;
        /**
         * The PrintStream to display the learning output
         */
        private PrintStream output;
        /**
         * The momentum vector
         */
        private Vector momentum;
        /**
         * The amount of data to proccess each step
         */
        private int bathSize;

        /**
         * The no args constructor
         */
        public NetworkTeacher() {
            this(new HashSet<>());
        }

        /**
         * A function to convert an array of inputs and outputs into test data
         * @param inputs The inputs
         * @param outputs The outputs
         * @return The test data
         */
        private Set<TestData> toTestDataSet(double[][] inputs, double[][] outputs) {
            if (inputs.length != outputs.length) {
                throw new InputMismatchException("You must submit the same amount of inputs as outputs! They come in pairs!");
            }
            Set<TestData> set = new HashSet<>();
            for (int i = 0; i < inputs.length; i++) {
                double[] in = inputs[i];
                double[] out = outputs[i];
                if (in.length != NeuralNet.this.getInputsCount() || out.length != getOutputsCount()) {
                    throw new InputMismatchException("You must enter " + getInputsCount() + " inputs and " + getOutputsCount() + " outputs!");
                }
                TestData data = new TestData(in, out) {
                    @Override
                    public boolean isCorrect(double[] actual) {
                        return false;
                    }
                };
                set.add(data);

            }
            return set;
        }

        /**
         * A function for changing the test data at runtime, this is okay beacuse the learning is "online" (stochastic) gradiant decent so the test data can safely change
         * @param data The test data
         */
        public void setTestData(Set<TestData> data) {
            this.testData = data;
            this.shuffle();
        }


        /**
         * A function to set the data to an array pair of inputs and outputs
         * @param inputs The inputs
         * @param outputs The outputs
         */
        public void setTestData(double[][] inputs, double[][] outputs) {
            this.setTestData(toTestDataSet(inputs, outputs));
        }

        /**
         * A constructor to create a network teacher from a set of test data
         * @param data The set of test data
         */
        public NetworkTeacher(Set<TestData> data) {
            this(data, data.size(), System.out);
        }

        /**
         * A constructor to create a network teacher
         * @param testData The test data to learn with
         * @param batchSize The amount of data per step
         * @param output The output stream to display the info
         */
        public NetworkTeacher(Set<TestData> testData, int batchSize, OutputStream output) {
            this.testData = testData;
            this.bathSize = batchSize;
            this.momentum = new Vector(getLayerSizes());
            this.output = new PrintStream(output);
            this.shuffle();
            this.output.format(outputPrefix, "#", "Step", "Momentum", "Cost", "ΔCost", "Accuracy", "Time");
            this.output.println();
        }

        /**
         * An accessor to set the oytput prefix
         * @param outputPrefix The output prefix
         */
        public void setOutputPrefix(String outputPrefix) {
            this.outputPrefix = outputPrefix;
        }

        /**
         * A function to change the output stream to display data
         * @param output The new output stream to display data
         */
        public void setOutput(PrintStream output) {
            this.output = output;
        }


        /**
         * A function for printing stuff to the output stream
         * @param out The stuff to print
         */
        private void print(String out) {
            this.output.println(" [NetworkTeacher] " + out);
        }

        /**
         * A function for shuffling the training data
         */
        private void shuffle() {
            shuffled = new ArrayList<>();
            List<TestData> data = new ArrayList<>(this.getTestData());
            Collections.shuffle(data);
            int size = testData.size();
            for (int i = 0; i < size / bathSize + 1; i++) {
                Set<TestData> set = new HashSet<>();
                for (int j = 0; j < bathSize && !data.isEmpty(); j++) {
                    set.add(data.get(0));
                    data.remove(0);
                }
                if (!set.isEmpty()) {
                    shuffled.add(set);
                }
            }
            if (bathSize < this.getTestData().size()) {
                print("Shuffled training data");
            }
        }

        /**
         *
         * @return The momentum facotr
         */
        public double getMomentumFactor() {
            return momentumFactor;
        }

        /**
         * Set the momentum factor
         * @param momentumFactor The new momentum factor
         */
        public void setMomentumFactor(double momentumFactor) {
            this.momentumFactor = momentumFactor;
        }

        /**
         * A function to get the next test data in the shuffed set
         * @return The new test data
         */
        private Set<TestData> getNextSet() {
            if (shuffled.isEmpty()) {
                shuffle();
            }
            return shuffled.remove(0);
        }

        /**
         * A function for adding a peice of test data at runtime
         * @param data The data to add
         */
        public void addTestData(TestData data) {
            this.testData.add(data);
        }

        /**
         * A function for getting all the training data
         * @return The training data
         */
        public Set<TestData> getTestData() {
            return testData;
        }

        /**
         *
         * @return The learning factor
         */
        public double getLearningFactor() {
            return learningFactor;
        }

        /**
         *
         * @param learningFactor The new learning factor
         */
        public void setLearningFactor(double learningFactor) {
            this.learningFactor = learningFactor;
        }

        /**
         * The update log from the last step
         */
        private UpdateLog lastLog = new UpdateLog(-1, getOutputsCount(), 0, 0, 0, 0, 0);

        /**
         * The l
         * @return The last step log
         */
        public UpdateLog getLastStepLog() {
            return lastLog;
        }


        /**
         * A function to learn the current data asyncronously in it's own thread
         * @param steps The amount of steps to preform
         */
        public void learnAsynchronously(int steps) {
            new Thread() {
                @Override
                public void run() {
                    for (int i = 0; i < steps; i++) {
                        gradientStep();
                    }
                }
            }.start();
        }

        /**
         * The core functon of the Network Teacher. Every time this function is called the network preforms a gradiant decent step.
         * @return The log with all the info about that gradiant step
         */
        public UpdateLog gradientStep() {
            long start = System.currentTimeMillis();
            double oldCost = getLastStepLog().getCost();
            double cost = 0;
            double accuracy = 0;
            Map<TestData, ActiveNetwork> outputs = new HashMap<>();
            Set<TestData> trainingSet = this.getNextSet();
            Vector vector = new Vector(getLayerSizes());
            for (TestData data : trainingSet) {
                ActiveNetwork active = getInstance(data.getInputs());
                outputs.put(data, active);
                cost += getCost(active, data);
                if (data.isCorrect(active.getOutputs())) {
                    accuracy++;
                }
                Vector desired = getGradientVector(active, data);
                vector.add(desired);
                //System.out.println(desired);
            }
            cost /= outputs.size();
            //accuracy = (double) correct / outputs.size();
            accuracy /= trainingSet.size();
            vector.multiply(getLearningFactor() / (double) trainingSet.size());
            double momentumLength = momentum.getMagnitude();
            momentum.multiply(momentumFactor);
            momentum.add(vector);
            add(momentum);
            double magnitude = vector.getMagnitude();
            long deltaTime = System.currentTimeMillis() - start;
            steps++;
            UpdateLog log = new UpdateLog(steps, cost, cost - oldCost, accuracy, magnitude, momentumLength, deltaTime);
            log.print();
            lastLog = log;
            return log;
        }

        /**
         * A function for getting the cost of a network call
         * @param active The active data
         * @param data The correct data
         * @return The cost / error of the network
         */
        private double getCost(ActiveNetwork active, TestData data) {
            double cost = 0;
            for (int i = 0; i < data.getOutputs().length; i++) {
                double diff = data.getOutputs()[i] - active.getOutputs()[i];
                cost += diff * diff;
            }
            return cost;
        }

        /**
         * A function for calculating the gradiant vector given a set of test data
         * @param active The active network
         * @param data The test data
         * @return The Network Vector
         */
        private Vector getGradientVector(ActiveNetwork active, TestData data) {
            //TODO
            double[][] error = getError(active, data);
            Vector vector = new Vector(getLayerSizes());
            for (int layer = 1; layer < getWeightings().length; layer++) {
                for (int id = 0; id < getWeightings()[layer].length; id++) {
                    double derivative = sigmoid_prime(inverse_sigmoid(active.getValue(layer, id)));
                    for (int id2 = 0; id2 < getWeightings()[layer - 1].length; id2++) {
                        double weightChange = error[layer][id] * derivative * active.getValue(layer - 1, id2);
                        vector.setWeight(layer, id, id2, weightChange);
                        //.getWeightings().getWeights().put(prior, weightChange);
                    }
                    double biasChange = error[layer][id] * derivative;
                    vector.setBias(layer, id, biasChange);
                    // map.put(neuron,.getWeightings());
                }
            }
            return vector;
        }

        /**
         * A function for getting the error of a network for each neuron
         * @param active The active network
         * @param data The test data
         * @return The error of the network double[layer][id] = error of that neuron
         */
        private double[][] getError(ActiveNetwork active, TestData data) {
            //TODO
            double[][] error = new double[getWeightings().length][];
            for (int layer = 0; layer < error.length; layer++) {
                error[layer] = new double[getWeightings()[layer].length];
            }
            for (int i = 0; i < error[error.length - 1].length; i++) {
                error[error.length - 1][i] = data.getOutputs()[i] - active.getValue(error.length - 1, i);
                //System.out.println(i + ": " + outputs[i] + " - " + getValue(error.length-1, i) + " = " + error[error.length-1][i]);
            }
            for (int layer = error.length - 2; layer >= 0; layer--) {
                for (int id = 0; id < getWeightings()[layer].length; id++) {
                    double diff = 0;
                    for (int target = 0; target < getWeightings()[layer + 1].length; target++) {
                        diff += getWeight(layer + 1, target, id) * error[layer + 1][target];
                    }
                    // System.out.println(diff);
                    error[layer][id] = diff;
                }
            }
            return error;
        }

        /**
         * The Update Log class
         */
        public class UpdateLog {
            /**
             * The cost after that step
             */
            private double cost;
            /**
             * How much the cost changed from the previous step
             */
            private double deltaCost;
            /**
             * The accuracy of the network after the step
             */
            private double accuracy;
            /**
             * The magnitude of the step vector
             */
            private double stepLength;
            /**
             * The magnitude of the momentum vector
             */
            private double momentumLength;
            /**
             * The amount of time it took to calculate the step
             */
            private long time;
            /**
             * The step number of this step
             */
            private int stepNumber;

            /**
             * The constructor
             * @param stepNumber step num
             * @param cost the cost
             * @param deltaCost the change in cost
             * @param accuracy The accuracy
             * @param stepLength the step magnitude
             * @param momentumLength The momentum magnitude
             * @param deltaTime The time it took to calculate this tep
             */
            public UpdateLog(int stepNumber, double cost, double deltaCost, double accuracy, double stepLength, double momentumLength, long deltaTime) {
                this.cost = cost;
                this.deltaCost = deltaCost;
                this.accuracy = accuracy;
                this.stepLength = stepLength;
                this.momentumLength = momentumLength;
                this.time = deltaTime;
                this.stepNumber = stepNumber;
            }

            public double getCost() {
                return cost;
            }

            public double getAccuracy() {
                return accuracy;
            }

            public double getDeltaCost() {
                return deltaCost;
            }

            public double getStepLength() {
                return stepLength;
            }

            public double getMomentumLength() {
                return momentumLength;
            }

            public long getTime() {
                return time;
            }

            public int getStepNumber() {
                return stepNumber;
            }

            /**
             * A function for printing this data to the learning stream
             */
            public void print() {
                this.print(output);
            }
            /**
             * A function for printing this data to a given output stream
             */

            public void print(PrintStream stream) {
                stream.println(toString());
            }

            /**
             * The function to represent this data as a string
             * @return The string of this log
             */
            @Override
            public String toString() {
                return String.format(outputPrefix, this.getStepNumber(), NumberFormat.getInstance().format(this.getStepLength()), NumberFormat.getInstance().format(momentumLength), NumberFormat.getInstance().format(this.getCost()), NumberFormat.getInstance().format(deltaCost), NumberFormat.getInstance().format(this.getAccuracy() * 100) + "%", +this.getTime() + " ms");
            }
        }
    }
}

/**
 * A class representing a vector in N dimensional space
 */
class Vector {
    /**
     * The weights of each neural connection
     */
    private double[][][] weightings;
    /**
     * The biases of each neuron
     */
    private double[][] biases;

    /**
     * The vector constructor based on weights and biases
     * @param weightings The weights
     * @param biases The biases
     */
    public Vector(double[][][] weightings, double[][] biases) {
        this.weightings = weightings;
        this.biases = biases;
    }

    /**
     * Construct a vector based on the layers neuron counts
     * @param layers the layers
     */
    public Vector(int... layers) {
        biases = new double[layers.length][];
        for (int i = 0; i < biases.length; i++) {
            biases[i] = new double[layers[i]];
        }
        weightings = new double[layers.length][][];
        for (int layer = 0; layer < weightings.length; layer++) {
            weightings[layer] = new double[layers[layer]][layer != 0 ? layers[layer - 1] : 0];
        }
    }

    /**
     * a quick pythagorist distance formula implemenation
     * @param distances The delta in each dimension
     * @return The magnitude
     */
    private static double distance(double... distances) {
        double magnitude = 0;
        for (int i = 0; i < distances.length; i++) {
            magnitude += distances[i] * distances[i];
        }
        magnitude = Math.sqrt(magnitude);
        return magnitude;
    }

    /**
     * A function to get the weights
     * @return The weightings
     */
    public double[][][] getWeightings() {
        return weightings;
    }


    /**
     * A function to get the biases
     * @return the biases in each dimension
     */
    public double[][] getBiases() {
        return biases;
    }

    /**
     * A function for setting the weight
     * @param layer The layer
     * @param neuron The neuron id
     * @param target the id of a neuron in the previous layer
     * @param weight the weight of the connection
     */
    public void setWeight(int layer, int neuron, int target, double weight) {
        weightings[layer][neuron][target] = weight;
    }

    /**
     * A function to get the weight of a connection
     * @param layer The layer
     * @param neuron The neuron id
     * @param target The id of a neuron in the previous layer
     * @return The weight of that connection
     */
    public double getWeight(int layer, int neuron, int target) {
        return weightings[layer][neuron][target];
    }

    /**
     * Set the bias of a neuron
     * @param layer The layer
     * @param neuron The id
     * @param bias the bias
     */
    public void setBias(int layer, int neuron, double bias) {
        biases[layer][neuron] = bias;
    }

    /**
     * Get the bias of a neuron
     * @param layer The layer id
     * @param neuron The neuron id
     * @return The bias of that connection
     */
    public double getBias(int layer, int neuron) {
        return biases[layer][neuron];
    }

    /**
     * A function for adding another vector to this one
     * @param vector The vector to add
     */
    public void add(Vector vector) {
        checkComparability(vector);
        for (int i = 0; i < this.weightings.length; i++) {
            for (int j = 0; j < this.weightings[i].length; j++) {
                for (int k = 0; k < this.weightings[i][j].length; k++) {
                    this.weightings[i][j][k] += vector.weightings[i][j][k];
                }
            }
        }
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                this.biases[i][j] += vector.biases[i][j];
            }
        }
    }

    /**
     * A function that checks if two neurons can be added together
     * @param vector The vector to compare against
     */
    private void checkComparability(Vector vector) {
        if (Arrays.compare(this.getLayerSizes(), vector.getLayerSizes()) != 0) {
            throw new InputMismatchException("Your two vectors must be the same size!");
        }
    }

    /**
     * A function to get the layer sizes
     * @return The layer sizes
     */
    public int[] getLayerSizes() {
        int[] arr = new int[biases.length];
        for (int i = 0; i < biases.length; i++) {
            arr[i] = biases[i].length;
        }
        return arr;
    }

    /**
     * A function for multiplying this vector by a constant
     * @param factor The constant
     */
    public void multiply(double factor) {
        for (int i = 0; i < this.weightings.length; i++) {
            for (int j = 0; j < this.weightings[i].length; j++) {
                for (int k = 0; k < this.weightings[i][j].length; k++) {
                    this.weightings[i][j][k] *= factor;
                }
            }
        }
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                this.biases[i][j] *= factor;
            }
        }
    }

    /**
     * A function to clone this vector
     * @return The clone of this vector
     */
    public Vector clone() {
        double[][][] changes = Arrays.copyOf(this.weightings, this.weightings.length);
        for (int i = 0; i < changes.length; i++) {
            changes[i] = Arrays.copyOf(this.weightings[i], this.weightings[i].length);
            for (int j = 0; j < changes[i].length; j++) {
                changes[i][j] = Arrays.copyOf(this.weightings[i][j], this.weightings[i][j].length);
            }
        }
        double[][] biases = Arrays.copyOf(this.biases, this.biases.length);
        for (int i = 0; i < biases.length; i++) {
            biases[i] = Arrays.copyOf(this.biases[i], this.biases[i].length);
        }
        return new Vector(changes, biases);
    }

    /**
     * A function to get the weight magnitude
     * @return The weight magnitude
     */
    public double getWeightMagnitude() {
        double magnitude = 0;
        for (int i = 0; i < this.weightings.length; i++) {
            for (int j = 0; j < this.weightings[i].length; j++) {
                for (int k = 0; k < this.weightings[i][j].length; k++) {
                    magnitude += weightings[i][j][k] * weightings[i][j][k];
                }
            }
        }
        magnitude = Math.sqrt(magnitude);
        return magnitude;
    }

    /**
     * The function to get the bias magnitude
     * @return The bias magnitude
     */
    public double getBiasMagnitude() {
        double magnitude = 0;
        for (int i = 0; i < biases.length; i++) {
            for (int j = 0; j < biases[i].length; j++) {
                magnitude += biases[i][j] * biases[i][j];
            }
        }
        magnitude = Math.sqrt(magnitude);
        return magnitude;
    }

    /**
     * The function to get the magnitude of this vector
     * @return The magnitude of this vector
     */
    public double getMagnitude() {
        return distance(getBiasMagnitude(), getWeightMagnitude());
    }
}
