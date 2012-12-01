#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.hpp>
#include <pcl/PointIndices.h>


using namespace std;
using Eigen::MatrixXd;


class Learner {
    int n;  // Number of superpixels
    int m;  // Number of features for each superpixel
    int k;  // Number of classes (labels)
    int l;  // Number of labeled superpixels so far
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    vector<pcl::PointIndices> superpixels;
    MatrixXd x;      // The n x m feature matrix
    MatrixXd sigma;  // The m x m feature lengthscale matrix
    MatrixXd W;      // The n x n weight matrix
    MatrixXd D;      // The n x n diagonal matrix of rowwise sums of W
    MatrixXd L;      // The n x n combinatorial Laplacian matrix, D - W
    MatrixXd f;      // The n x k matrix of known labels
    MatrixXd f_u;    // The u x k matrix of soft labels for unlabeled data
    
    void compute_features(void);
    
    public:
    void Learner(int, int, pcl::PointCloud<pcl::PointXYZRGB>&,
                      vector<pcl::PointIndices>&);
    void compute_weight_matrices(void);
    int update_label(int, MatrixXd&);
    void compute_harmonic_solution(void);
    int most_uncertain_superpixel(void);
    int least_uncertain_superpixel(void);
    void learn_weights(void);
};


void Learner::Learner(int classes,
                      pcl::PointCloud<pcl::PointXYZRGB>& point_cloud,
                      vector<pcl::PointIndices>& clusters) {
    n = clusters.size();
    m = 6;
    k = classes;
    l = 0;
    cloud = point_cloud;
    superpixels = clusters;
    
    MatrixXd x(n, m);
    compute_features();
    
    MatrixXd sigma(m, m);
    sigma << MatrixXd::Identity(m, m);
    
    MatrixXd W(n, n);
    MatrixXd D(n, n);
    MatrixXd L(n, n);
    compute_weight_matrices();
    
    MatrixXd f(n, k);
    f << MatrixXd::Zero(n, k);
}


void Learner::compute_features() {
    // For each superpixel, populate its feature vector with spatial, color,
    // and texture information.
    pcl::PointIndices cluster;
    pcl::PointXYZRGB point;
    int cluster_size, x_sum, y_sum, z_sum;
    int r_sum, g_sum, b_sum;
    for (i = 0; i < n; i++) {
        cluster = superpixel[i];
        cluster_size = cluster.size();
        x_sum = y_sum = z_sum = 0;
        r_sum = g_sum = b_sum = 0;
        for (j = 0; j < cluster_size; j++) {
            point = cluster[j];
            x_sum += point.x;
            y_sum += point.y;
            z_sum += point.z;
            r_sum += point.r;
            g_sum += point.g;
            b_sum += point.b;
        }
        x(i, 0) = (double) x_sum / cluster_size;
        x(i, 1) = (double) y_sum / cluster_size;
        x(i, 2) = (double) z_sum / cluster_size;
        x(i, 3) = (double) r_sum / cluster_size;
        x(i, 4) = (double) g_sum / cluster_size;
        x(i, 5) = (double) b_sum / cluster_size;
    }
}


void Learner::compute_weight_matrices() {
    // Build weight matrix W of similarities between each superpixel pair.
    // Similarity is given by exp(-g^T * sigma * g) where g = x_i - x_j.
    MatrixXd g(m, 1);
    double weight;
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            g.col(1) = x.row(i) - x.row(j);
            weight = exp(-g.transpose() * sigma * g);
            W(i, j) = weight;
            if (i != j)
                W(j, i) = weight;
        }
    }
    
    // Compute the diagonal matrix D of row sums of W
    D = W.rowwise().sum().asDiagonal();
    
    // Compute the combinatorial Laplacian L = D - W
    L = D - W;
}


int Learner::update_label(int index, MatrixXd& label) {
    // Return error code -1 if the given index has already been set
    if (index < l)
        return -1;
    
    // Update the label vector f with the new label
    f.row(l) = label;
    
    // Swap rows so that labeled data remains in the upper left
    W.row(index).swap(W.row(l));
    L.row(index).swap(L.row(l));
    x.row(index).swap(x.row(l));
    pcl::PointIndices temp1 = superpixels[index];
    superpixels[index] = superpixels[l];
    superpixels[l] = temp1;
    
    // Increment the number of labeled superpixels and return 0 for success
    l++;
    return 0;
}


void Learner::compute_harmonic_solution() {
    int u = n - l;
    MatrixXd f_u(u, k);
    f_u = L.bottomRightCorner(u, u).inverse() * W.bottomLeftCorner(u, l) * f.head(l);
}

 
int Learner::least_uncertain_superpixel() {
    // Find the k minimum entropy rows (maximun of sum of log(p))
    MatrixXf::Index index;
    f_u.array().log().rowwise().sum().maxCoeff(&index);
    return (int) index;
}


int Learner::most_uncertain_superpixel() {
    // Find the maximum entropy row (minimum of sum of log(p))
    MatrixXf::Index index;
    f_u.array().log().rowwise().sum().minCoeff(&index);
    return (int) index;
}


void Learner::learn_weights() {
    // Set up X^T * diag(sigma) = y, where
    // X is (m x l^2) and has [g_ij(1)^2, ..., g_ij(m)^2]^T in each column, and
    // y is (l^2 x 1) and has 0 if f_i = f_j and infinity if f_i != f_j.
    // Solution is diag(sigma) = (X^T * X)^-1 *X^T * y.
    
}














