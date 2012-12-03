#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include "segment.cpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array3d;


const double ZERO = 0.00000000001;


class Learner {
    int n;  // Number of superpixels
    int m;  // Number of features for each superpixel
    int k;  // Number of classes (labels)
    int l;  // Number of labeled superpixels so far
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
    vector<pcl::PointIndices> superpixels;
    MatrixXd x;      // The n x m feature matrix
    MatrixXd sigma;  // The m x m feature lengthscale matrix
    MatrixXd W;      // The n x n weight matrix
    MatrixXd D;      // The n x n diagonal matrix of rowwise sums of W
//    MatrixXd L;      // The n x n combinatorial Laplacian matrix, D - W
    MatrixXd P;
    MatrixXd f;      // The n x k matrix of known labels
    MatrixXd f_u;    // The u x k matrix of soft labels for unlabeled data
    
    void compute_features(void);
    
    public:
    Learner(int, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr,
                      vector<pcl::PointIndices>);
    void compute_weight_matrices(void);
    int update_label(int, VectorXd);
    void compute_harmonic_solution(void);
    int most_uncertain_superpixel(void);
    int least_uncertain_superpixel(void);
    void learn_weights(void);
    void self_train(int);
};


Learner::Learner(int classes,
                 pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud,
                 vector<pcl::PointIndices> clusters) {
    n = clusters.size();
    m = 6;
    k = classes;
    l = 0;
    cloud = point_cloud;
    superpixels = clusters;
    
    x = MatrixXd(n, m);
    compute_features();
    
    sigma = MatrixXd(m, m);
    sigma << MatrixXd::Identity(m, m);
    
    W = MatrixXd(n, n);
    D = MatrixXd(n, n);
//    L = MatrixXd(n, n);
    P = MatrixXd(n, n);
    compute_weight_matrices();
    
    f = MatrixXd(n, k);
    f << MatrixXd::Zero(n, k);
    
//    cout << "x:" << endl << x << endl;
//    cout << "W:" << endl << W << endl;
//    cout << "D:" << endl << D << endl;
//    cout << "L:" << endl << L << endl;
//    cout << "sigma:" << endl << sigma << endl;
//    cout << "f:" << endl << f << endl;
    
}


void Learner::compute_features() {
    pcl::PointIndices cluster;
    pcl::PointXYZRGBA point;
    int cluster_size;
    double x_sum, y_sum, z_sum;
    Array3d color_sum;
    
    // Compute spatial and color features for each superpixel
    for (int i = 0; i < n; i++) {
        cluster = superpixels[i];
        cluster_size = cluster.indices.size();
        x_sum = y_sum = z_sum = 0;
        color_sum << 0, 0, 0;
        
        // For each point in the superpixel
        for (int j = 0; j < cluster_size; j++) {
            
            // Get the spatial coordinates
            point = cloud->points[cluster.indices[j]];
            x_sum += point.x;
            y_sum += point.y;
            z_sum += point.z;
            
            // Get the color
            color_sum += point.getRGBVector3i().cast<double>().array() / 255.0;
        }
        
        // Compute the average location and color for this superpixel
        double size = (double) cluster_size;
        x(i, 0) = x_sum / size;
        x(i, 1) = y_sum / size;
        x(i, 2) = z_sum / size;
        x.block(i, 3, 1, 3) = (color_sum / size).matrix().transpose();
    }
}


void Learner::compute_weight_matrices() {
    // Build weight matrix W of similarities between each superpixel pair.
    // Similarity is given by exp(-g^T * sigma * g) where g = x_i - x_j.
    VectorXd g(m);
    double weight;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            g = x.row(i) - x.row(j);
            weight = exp((-g.transpose() * sigma * g).value());
            W(i, j) = weight;
            if (i != j)
                W(j, i) = weight;
        }
    }
    // Compute the diagonal matrix D of row sums of W
    D = W.rowwise().sum().asDiagonal();
    
    // Compute the combinatorial Laplacian L = D - W
//    L = D - W;  
    P = D.inverse() * W;
}


int Learner::update_label(int index, VectorXd label) {
    // Return error code -1 if the given index has already been set
    if (index < l)
        return -1;
    
    // Update the label vector f with the new label
    f.row(index) = label.transpose();
//    f.row(l) = label.transpose();
    
    // Swap rows so that labeled data remains in the upper left
    f.row(index).swap(f.row(l));
    W.row(index).swap(W.row(l));
//    L.row(index).swap(L.row(l));
    x.row(index).swap(x.row(l));
    pcl::PointIndices temp1 = superpixels[index];
    superpixels[index] = superpixels[l];
    superpixels[l] = temp1;
    
//    cout << "x:" << endl << x << endl;
//    cout << "f:" << endl << f << endl;
    
    // Increment the number of labeled superpixels and return 0 for success
    l++;
    return 0;
}


void Learner::compute_harmonic_solution() {
    int u = n - l;
    f_u = MatrixXd(u, k);
//    f_u = L.bottomRightCorner(u, u).inverse() * W.bottomLeftCorner(u, l) * f.topRows(l);
    f_u = (MatrixXd::Identity(u, u) - P.bottomRightCorner(u, u)).inverse() * P.bottomLeftCorner(u, l) * f.topRows(l);
    
    f.bottomRows(u) = f_u;
    cout << "f:" << endl << f << endl << endl;
}

 
int Learner::least_uncertain_superpixel() {
    // Find the minimum entropy row (min of sum of -p*log(p))
    // Returns the index into superpixels, not into f_u
    MatrixXd::Index index;
    (-f_u.array() * (ZERO + f_u.array()).log()).rowwise().sum().minCoeff(&index);
    
//    cout << "Certain: " << f_u.row(index) << endl;
    
    return (int) index + l;
}


int Learner::most_uncertain_superpixel() {
    // Find the maximum entropy row (max of sum of -p*log(p))
    // Returns the index into superpixels, not into f_u
    MatrixXd::Index index;
    (-f_u.array() * (ZERO + f_u.array()).log()).rowwise().sum().maxCoeff(&index);
    
//    cout << "Unertain: " << f_u.row(index) << endl;
    
    return (int) index + l;
}


void Learner::learn_weights() {
    // Set up X^T * diag(sigma) = y, where
    // X is (l^2 x m) and has [g_ij(1)^2, ..., g_ij(m)^2] in each row, and
    // y is (l^2 x 1) and has 0 if f_i = f_j and infinity if f_i != f_j.
    // Solution is diag(sigma) = (X^T * X)^-1 *X^T * y.
    
    int num_constraints = (l * (l - 1)) / 2;
    MatrixXd X(num_constraints, m);
    VectorXd y(num_constraints);
    
    int index = 0;
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < i; j++) {
            VectorXd g = x.row(i) - x.row(j);
            X.row(index) = g.array() * g.array();
            if ((f.row(i).array() == f.row(j).array()).all())
                y(index) = 0;
            else
                y(index) = 100;
            index++;
        }
    }
    sigma.diagonal() = (X.transpose() * X).inverse() * X.transpose() * y;
//    cout << "sigma:" << endl << sigma << endl;
}


void Learner::self_train(int max_iters) {
    int really_max_iters = n - l;
    for (int added = 0; added < max_iters && added < really_max_iters; added++) {
        
        int i = least_uncertain_superpixel();
        if ((-f_u.row(i-l).array() * (ZERO + f_u.row(i-l).array()).log()).sum() > 0.6)
            break;
        
        MatrixXd::Index max_index;
        f_u.row(i-l).maxCoeff(&max_index);
        
        VectorXd label(k);
        label = VectorXd::Zero(k);
        label(max_index) = 1.0;
        
        update_label(i, label);
        compute_harmonic_solution();
    }
}


int main (int argc, char** argv) {
       
//    test_segmentation();
    
    // Read in point cloud and downsample
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = read_pcd();
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr small_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    downsample(cloud, small_cloud);
    
    // Extract Euclidean clusters
    std::vector<pcl::PointIndices> clusters = euclidean_clusters(small_cloud);
    
    // Create the active learning object
    int num_classes = 4;
    Learner learner(num_classes, small_cloud, clusters);
    
    // Update the labels of four of the superpixels
    VectorXd label(num_classes);
    label << 1.0, 0.0, 0.0, 0.0;
    learner.update_label(20, label);
    label << 0.0, 1.0, 0.0, 0.0;
    learner.update_label(10, label);
    label << 0.0, 1.0, 0.0, 0.0;
    learner.update_label(15, label);
    label << 0.0, 0.0, 0.0, 1.0;
    learner.update_label(5, label);
    
    // Learn the feature weights and recompute weight matrices
//    learner.learn_weights();
//    learner.compute_weight_matrices();
    
    // Propagate labels to unlabeled superpixels
    learner.compute_harmonic_solution();
    
    // Self-train
    learner.self_train(10);
    
    // Choose next superpixel for human labeling
//    int uncertain = learner.most_uncertain_superpixel();
}











