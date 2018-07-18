

import me.izzyk.neuralnet.NeuralNet;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import static java.lang.String.format;

public class HandwrittenDigitsNeuralNetwork {
    public static void main(String[] args) throws IOException {
        File file = new File("Network/network.txt");
        NeuralNet net;
        File learning = new File("Network/learning.txt");
        if (file.exists()) {
            net = new NeuralNet(file);
        } else {
            net = new NeuralNet(28 * 28, 28, 18, 10);
        }
        NeuralNet.NetworkTeacher teacher = net.new NetworkTeacher(readMinstDatabase(net), 1000, new FileOutputStream(learning));
        System.out.println(net.getPathwayCount());
        for (int i = 0; true; i++) {
            teacher.gradientStep();
            if (i % 50 == 0) {
                net.save(file);
                System.out.println("Updated Network File");
            }
        }
    }

    public static final int MINST_DATABASE_SIZE =  MnistReader.getLabels("TrainingData/train-labels.idx1-ubyte").length;

    public static Set<NeuralNet.TestData> readMinstDatabase(NeuralNet net) {
        return readMinstDatabase(net, MINST_DATABASE_SIZE, "TrainingData/train-images.idx3-ubyte", "TrainingData/train-labels.idx1-ubyte");
    }
    public static Set<NeuralNet.TestData> readMinstDatabase(NeuralNet net, int size, String imagesFile, String labelFile) {
        List<int[][]> images = MnistReader.getImages(imagesFile);
        int[] labels = MnistReader.getLabels(labelFile);
        ArrayList<Integer> random = new ArrayList<>();
        for (int i = 0; i < images.size(); i++) {
            random.add(i);
        }
        Collections.shuffle(random);
        Set<NeuralNet.TestData> set = new HashSet<>();
        for (int i = 0; i < labels.length && i < size; i++) {
            int index = random.get(i);
            double[] output = new double[10];
            for (int n = 0; n < 10; n++) {
                output[n] = labels[index] == n ? 1D : 0D;
            }
            double[] input = new double[28 * 28];
            int[][] image = images.get(index);
            for (int r = 0; r < image.length; r++) {
                for (int c = 0; c < image[r].length; c++) {
                    int n = r + 28 * c;
                    input[n] =(double) image[r][c] / 255.0;
                }
            }
            set.add(net.new TestData(input, output){
                @Override
                public boolean isCorrect(double[] actual) {
                    return HandwrittenDigitsNeuralNetwork.isCorrect(actual, this.getOutputs());
                }
            });
        }
        return set;
    }

    public static boolean isCorrect(double[] actual, double[] expected) {
        int highestActual = 0;
        int highestExpected = 0;
        for (int i = 1; i < actual.length; i++) {
            if (actual[i] > actual[highestActual]) {
                highestActual = i;
            }
            if (expected[i] > expected[highestExpected]) {
                highestExpected = i;
            }
        }
        return highestActual == highestExpected;
    }
}

/**
 * This class is from the internet
 * https://github.com/jeffgriffith/mnist-reader
 */
class MnistReader {
    public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
    public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;

    public static int[] getLabels(String infile) {

        ByteBuffer bb = loadFileToByteBuffer(infile);

        assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());

        int numLabels = bb.getInt();
        int[] labels = new int[numLabels];

        for (int i = 0; i < numLabels; ++i)
            labels[i] = bb.get() & 0xFF; // To unsigned

        return labels;
    }

    public static List<int[][]> getImages(String infile) {
        ByteBuffer bb = loadFileToByteBuffer(infile);

        assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());

        int numImages = bb.getInt();
        int numRows = bb.getInt();
        int numColumns = bb.getInt();
        List<int[][]> images = new ArrayList<>();

        for (int i = 0; i < numImages; i++)
            images.add(readImage(numRows, numColumns, bb));

        return images;
    }

    private static int[][] readImage(int numRows, int numCols, ByteBuffer bb) {
        int[][] image = new int[numRows][];
        for (int row = 0; row < numRows; row++)
            image[row] = readRow(numCols, bb);
        return image;
    }

    private static int[] readRow(int numCols, ByteBuffer bb) {
        int[] row = new int[numCols];
        for (int col = 0; col < numCols; ++col)
            row[col] = bb.get() & 0xFF; // To unsigned
        return row;
    }

    public static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
        if (expectedMagicNumber != magicNumber) {
            switch (expectedMagicNumber) {
                case LABEL_FILE_MAGIC_NUMBER:
                    throw new RuntimeException("This is not a label file.");
                case IMAGE_FILE_MAGIC_NUMBER:
                    throw new RuntimeException("This is not an image file.");
                default:
                    throw new RuntimeException(
                            format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
            }
        }
    }

    /*******
     * Just very ugly utilities below here. Best not to subject yourself to
     * them. ;-)
     ******/

    public static ByteBuffer loadFileToByteBuffer(String infile) {
        return ByteBuffer.wrap(loadFile(infile));
    }

    public static byte[] loadFile(String infile) {
        try {
            RandomAccessFile f = new RandomAccessFile(infile, "r");
            FileChannel chan = f.getChannel();
            long fileSize = chan.size();
            ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
            chan.read(bb);
            bb.flip();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            for (int i = 0; i < fileSize; i++)
                baos.write(bb.get());
            chan.close();
            f.close();
            return baos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String renderImage(int[][] image) {
        StringBuffer sb = new StringBuffer();

        for (int row = 0; row < image.length; row++) {
            sb.append("|");
            for (int col = 0; col < image[row].length; col++) {
                int pixelVal = image[row][col];
                if (pixelVal == 0)
                    sb.append(" ");
                else if (pixelVal < 256 / 3)
                    sb.append(".");
                else if (pixelVal < 2 * (256 / 3))
                    sb.append("x");
                else
                    sb.append("X");
            }
            sb.append("|\n");
        }

        return sb.toString();
    }

    public static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.append(s);
        return sb.toString();
    }
}
