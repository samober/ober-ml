package com.ober.ml.cw;

import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.HelpFormatter;

public class App {
    
    public static void main( String[] args ) {
        CommandLineParser clParser = new BasicParser();
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("graph file")
            .hasArg()
            .withDescription("input graph binary file")
            .isRequired()
            .create("graph"));
        options.addOption(OptionBuilder.withArgName("output file")
            .hasArg()
            .withDescription("path for output cluster file")
            .isRequired()
            .create("output"));
        options.addOption(OptionBuilder.withArgName("max edges")
            .hasArg()
            .withDescription("maximum number of edges to consider for each node")
            .create("max_edges"));
        options.addOption(OptionBuilder.withArgName("max connectivity")
            .hasArg()
            .withDescription("maximum number of edges each subnode can have in an ego network")
            .create("max_connectivity"));
        options.addOption(OptionBuilder.withArgName("max iterations")
            .hasArg()
            .withDescription("maximum number of times to run chinese whispers")
            .create("max_iterations"));
        options.addOption(OptionBuilder.withArgName("min cluster size")
            .hasArg()
            .withDescription("minimum size for each cluster")
            .create("min_cluster"));
        options.addOption(OptionBuilder.withArgName("num workers")
            .hasArg()
            .withDescription("number of worker threads")
            .create("num_workers"));
            
        CommandLine cl = null;
        boolean success = false;
        
        try {
            cl = clParser.parse(options, args);
            success = true;
        } catch (ParseException e) {
            System.out.println(e.getMessage());
        }
        
        if (!success) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("CWD", options, true);
            System.exit(1);
        }

        String filename = cl.getOptionValue("graph");
        String output = cl.getOptionValue("output");
        
        int maxEdges = cl.hasOption("max_edges") ? Integer.parseInt(cl.getOptionValue("max_edges")) : 200;
        int maxConnectivity = cl.hasOption("max_connectivity") ? Integer.parseInt(cl.getOptionValue("max_connectivity")) : 200;
        int maxIterations = cl.hasOption("max_iterations") ? Integer.parseInt(cl.getOptionValue("max_iterations")) : 100;
        int minCluster = cl.hasOption("min_cluster") ? Integer.parseInt(cl.getOptionValue("min_cluster")) : 5;
        int numWorkers = cl.hasOption("num_workers") ? Integer.parseInt(cl.getOptionValue("num_workers")) : 4;
		
		WSI wsi = new WSI(filename, output, maxEdges, maxConnectivity, maxIterations, minCluster, numWorkers);
        wsi.loadGraph();
        wsi.calculateSenses();
    }
    
}
