package com.webgraph;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class Main {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java Main <graphPath> <binaryFilePath>");
            return;
        }

        // 指定图的路径
        String graphPath = args[0];

        // 指定输出的二进制文件路径
        String binaryFilePath = args[1];

        System.out.println("Input: " + graphPath);
        System.out.println("Output: " + binaryFilePath);

        // 加载 .graph 文件
        ImmutableGraph graph;
        try {
            graph = ImmutableGraph.loadSequential(graphPath);

            // 获取图的节点数和边数
            NodeIterator nodeIterator = graph.nodeIterator();
            int total = 0;
            try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(binaryFilePath))) {
                ByteBuffer buffer = ByteBuffer.allocate(8192);
                // 缓冲区大小为 8 KB

                // int nodeId;
                while (nodeIterator.hasNext()) {
                    int u = nodeIterator.nextInt();

                    int[] successors = nodeIterator.successorArray();
                    int outdegree = nodeIterator.outdegree();
                    // nodeId = u;
                    // if (nodeId % 1000 == 0) {
                    // System.out.println(nodeId + " " + outdegree);
                    // }
                    for (int i = 0; i < outdegree; i++) {
                        int v = successors[i];
                        if (buffer.remaining() < Long.BYTES * 2) {
                            dos.write(buffer.array(), 0, buffer.position());
                            buffer.clear();
                        }
                        buffer.putLong(u);
                        buffer.putLong(v);
                    }
                    total += outdegree;
                }

                // 写入剩余数据
                if (buffer.position() > 0) {
                    dos.write(buffer.array(), 0, buffer.position());
                }

                System.out.println("Total vertices: " + graph.numNodes());
                System.out.println("Total edges: " + total);
            } catch (IOException e) {
                System.err.println("Error writing to binary file: " + e.getMessage());
            }
        } catch (Exception e) {
            System.err.println("Error loading graph: " + e.getMessage());
        }
    }
}

// java -cp webgraph-3.5.1.jar it.unimi.dsi.webgraph.BVGraph -o -m 1 basename-hc
// basename-fast
// // java -cp
// .:/home/LiJB/cuda_project/TC-compare-V100/dataset_app/web_dataset/webbase-2001/graph.jar
// it.unimi.dsi.webgraph.BVGraph