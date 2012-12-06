#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/common/transforms.h>
#include <cv.h>
#include <highgui.h>
#include "segment.cpp"

using namespace std;
using namespace cv;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Matrix4f;
using Eigen::Vector4f;
using Eigen::Vector3f;
using Eigen::Vector3i;
using Eigen::VectorXd;
using Eigen::Array3d;

typedef Vec<unsigned char, 3> Vec3u;


const double ZERO = 0.00000000001;

bool lowest_z_last(pcl::PointXYZRGBA p1, pcl::PointXYZRGBA p2) {
    return (p1.z > p2.z);
}


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
    VectorXd entropies; // The u-dimensional vector of entropies based on f_u
    vector<int> sp_indices; // Keeps track of swapped indices
    
    void compute_features(void);
    
    public:
    Learner(int, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr,
                      vector<pcl::PointIndices>);
    void compute_weight_matrices(void);
    int update_label(int, VectorXd);
    void compute_harmonic_solution(void);
    vector<int> most_uncertain_superpixels(int);
    vector<int> least_uncertain_superpixels(int);
    vector<int> random_superpixels(int);
    void learn_weights(void);
    void self_train(int, double);
    int interactive_learn(int, int, bool, bool, bool);
    MatrixXd get_labels();
};


Learner::Learner(int classes,
                 pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud,
                 vector<pcl::PointIndices> clusters) {
    n = clusters.size();
    m = 3;
    k = classes;
    l = 0;
    cloud = point_cloud;
    superpixels = clusters;
    
    x = MatrixXd(n, m);
    compute_features();
    
    sigma = MatrixXd(m, m);
    sigma << 0.5 * MatrixXd::Identity(m, m);
    
    W = MatrixXd(n, n);
    D = MatrixXd(n, n);
//    L = MatrixXd(n, n);
    P = MatrixXd(n, n);
    compute_weight_matrices();
    
    f = MatrixXd(n, k);
    f << MatrixXd::Zero(n, k);
    
    for (int i = 0; i < n; i++)
        sp_indices.push_back(i);
    
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
//            color_sum += point.getRGBVector3i().cast<double>().array() / 255.0;
        }
        
        // Compute the average location and color for this superpixel
        double size = (double) cluster_size;
        x(i, 0) = x_sum / size;
        x(i, 1) = y_sum / size;
        x(i, 2) = z_sum / size;
//        x.block(i, 3, 1, 3) = (color_sum / size).matrix().transpose();
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
    
    // Compute the matrix P used in computing
    P = D.inverse() * W;
}


int Learner::update_label(int index, VectorXd label) {
    // Return error code -1 if the given index has already been set
    if (index < l)
        return -1;
    
    // Update the label vector f with the new label
    f.row(index) = label.transpose();
    
    // Swap rows so that labeled data remains in the upper left
    f.row(index).swap(f.row(l));
    W.row(index).swap(W.row(l));
    x.row(index).swap(x.row(l));
    pcl::PointIndices temp1 = superpixels[index];
    superpixels[index] = superpixels[l];
    superpixels[l] = temp1;
    int temp2 = sp_indices[index];
    sp_indices[index] = sp_indices[l];
    sp_indices[l] = temp2;
    
    // Increment the number of labeled superpixels and return 0 for success
    l++;
    return 0;
}


void Learner::compute_harmonic_solution() {
    int u = n - l;
    f_u = MatrixXd(u, k);
    f_u = (MatrixXd::Identity(u, u) - P.bottomRightCorner(u, u)).inverse() * P.bottomLeftCorner(u, l) * f.topRows(l);
    
    f.bottomRows(u) = f_u;
    cout << "f:" << endl << f << endl << endl;
    
    entropies = VectorXd(u);
    entropies = (-f_u.array() * (ZERO + f_u.array()).log()).rowwise().sum();
}

 
vector<int> Learner::least_uncertain_superpixels(int count) {
    // Returns the indices of the superpixels with smallest entropy
    
    entropies = -entropies;
    vector<int> indices = most_uncertain_superpixels(count);
    entropies = -entropies;
    
    return indices;
}


vector<int> Learner::most_uncertain_superpixels(int count) {
    // Returns the indices of the superpixels with largest entropy
    
    vector<int> indices;
    MatrixXd::Index index;
    
    if (count > n - l)
        count = n - l;
    if (count == 0)
        return indices;
    
    double previous_max = entropies.maxCoeff(&index);
    indices.push_back((int) index + l);
    
    while (indices.size() < count) {
        double max_value = -1000.0;
        int max_index = 0;
        for (int i = 0; i < entropies.size(); i++) {
            if (entropies(i) < previous_max && entropies(i) > max_value) {
                max_index = i;
                max_value = entropies(i);
            }
        }
        indices.push_back(max_index + l);
        previous_max = max_value;
    }
    
    return indices;
}


vector<int> Learner::random_superpixels(int count) {
    // Returns the indices of random unlabeled superpixels
    
    vector<int> indices;
    
    if (count > n - l)
        count = n - l;
    if (count == 0)
        return indices;
    
    vector<int> all_indices;
    for (int i = l; i < n; i++)
        all_indices.push_back(i);
    
    srand(time(NULL));
    while (indices.size() < count) {
        int random = rand() % all_indices.size();
        indices.push_back(all_indices[random]);
        all_indices.erase(all_indices.begin() + random);
    }
    
    return indices;
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
    cout << "sigma:" << endl << sigma << endl;
}


void Learner::self_train(int max_iters, double threshold) {
    if (max_iters > n - l)
        max_iters = n - l;
    
    for (int added = 0; added < max_iters; added++) {
        
        int i = least_uncertain_superpixels(1)[0];
        if (entropies(i - l) > threshold)
            break;
        
        MatrixXd::Index max_index;
        f_u.row(i - l).maxCoeff(&max_index);
        
        VectorXd label(k);
        label = VectorXd::Zero(k);
        label(max_index) = 1.0;
        
        update_label(i, label);
        compute_harmonic_solution();
    }
}


int Learner::interactive_learn(int max_iters,
                                int labels_per_iter,
                                bool use_self_training,
                                bool use_active,
                                bool use_weight_learning) {

    // Set up camera parameters and perspective projection matrix
    float width = 640.0;
    float height = 480.0;
    float alpha = 500.0;
    float beta = 500.0;
    float x_0 = width / 2.0;
    float y_0 = height / 2.0;
    
    Matrix<float, 3, 4> projection;
    projection << alpha, 0,    x_0,  0, 
                  0,     beta, y_0,  0,
                  0,     0,    1,    0;
    
    // Initialize things
    Mat image((int) height, (int) width, CV_8UC3, Scalar(0, 0, 0));
    namedWindow("Label this image", CV_WINDOW_AUTOSIZE);
    
    pcl::PointXYZRGBA point;
    Vector4f P;
    Vector3f p;
    Vector3i rgb;
    vector<int> indices;
    
    // Sort the point cloud by distance to camera so projection works correctly
    vector<pcl::PointXYZRGBA, Eigen::aligned_allocator<pcl::PointXYZRGBA> > sorted = cloud->points;
    sort(sorted.begin(), sorted.end(), lowest_z_last);
    
    // Main loop
    int user_labels = 0;
    for (int iter = 0; iter < max_iters; iter++) {
        
        // Find the next superpixels to label
        if (use_active) {
            compute_harmonic_solution();
            indices = most_uncertain_superpixels(labels_per_iter);
        }
        else {
            indices = random_superpixels(labels_per_iter);
        }
        if (indices.size() == 0)
            return user_labels;
    
        // For each superpixel to be labeled
        for (int sp = 0; sp < indices.size(); sp++) {
            pcl::PointIndices cluster = superpixels[indices[sp]];
            cout << endl << indices[sp] << ": " << f.row(indices[sp]) << endl << endl;
            
            // Iterate over all points in the sorted cloud
            for (int i = 0; i < sorted.size(); i++) {
                point = sorted[i];
                P << point.x, point.y, point.z, 1.0;
                p = (projection * P).array() / P(2);
                int row = (int) p(1);
                int col = (int) p(0);
                if (row >= 0 && row < height && col >= 0 and col < width) {
                    rgb = point.getRGBVector3i();
                    for (int j = 0; j < 3; j++)
                        image.at<Vec3u>(row, col)[j] = (unsigned char) (rgb(2 - j) / 8);
                }
            }
            
            // Iterate over the points in the uncertain superpixels
            for (int j = 0; j < cluster.indices.size(); j++) {
                point = cloud->points[cluster.indices[j]];
                P << point.x, point.y, point.z, 1.0;
                p = (projection * P).array() / P(2);
                int row = (int) p(1);
                int col = (int) p(0);
                if (row >= 0 && row < height && col >= 0 and col < width) {
                    rgb = point.getRGBVector3i();
                    for (int j = 0; j < 3; j++)
                        image.at<Vec3u>(row, col)[j] = (unsigned char) rgb(2 - j);
                }
            }
            
            // Display the image and wait for a response
            imshow("Label this image", image);
//            imwrite("image.png", image);
            
            char c = 'a';
            while (c != 'q' && (c < '0' || c > '9'))
                c = waitKey(0);
            
            if (c == 'q')
                return user_labels;
            
            // Assume the key press was a label
            VectorXd label(k);
            label << VectorXd::Zero(k);
            label(c - '0') = 1.0;
            update_label(indices[sp], label);
            user_labels++;
        }
        
        // Before going on to the next iteration
        if (use_weight_learning && (l*(l-1))/2 >= m) {
            learn_weights();
            compute_weight_matrices();
        }
        if (use_self_training && l > k) {
            compute_harmonic_solution();
            self_train(20, 0.4);
        }
    }
    
    return user_labels;
}


MatrixXd Learner::get_labels() {
    // Get the label matrix in a canonical order
    MatrixXd new_f(n, k);
    for (int i = 0; i < n; i++)
        new_f.row(sp_indices[i]) = f.row(i);
    return new_f;
}


void evaluate_algorithms() {
    
    int num_classes = 4;
    
    // Read the point cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = read_pcd();
    
    // Segment the point cloud
    std::vector<pcl::PointIndices> clusters = euclidean_clusters(cloud);
//    std::vector<pcl::PointIndices> clusters = region_growing_segmentation(cloud);
    
    // Create the learners
    Learner learner1(num_classes, cloud, clusters);
    Learner learner2(num_classes, cloud, clusters);
    Learner learner3(num_classes, cloud, clusters);
    
    // Random learner
    int user_labels1 = learner1.interactive_learn(1000, 1, false, false, false);
    MatrixXd ground_truth = learner1.get_labels();
    double accuracy1 = 1.0;
    
    // Self-training only
    int user_labels2 = learner2.interactive_learn(1000, 1, true, false, false);
    MatrixXd labels2 = learner2.get_labels();
    double accuracy2 = (labels2.array() * ground_truth.array()).sum() / labels2.rows();
    
    // Self-training + active
    int user_labels3 = learner3.interactive_learn(1000, 1, true, true, false);
    MatrixXd labels3 = learner3.get_labels();
    double accuracy3 = (labels3.array() * ground_truth.array()).sum() / labels3.rows();
    
    cout << endl;
    
    cout << "Random" << endl
         << "User labels: " << user_labels1 << endl
         << "Accuracy:    " << accuracy1 << endl
         << endl;
    
    cout << "Self-training only" << endl
         << "User labels: " << user_labels2 << endl
         << "Accuracy:    " << accuracy2 << endl
         << endl;
    
    cout << "Self-training + active" << endl
         << "User labels: " << user_labels3 << endl
         << "Accuracy:    " << accuracy3 << endl
         << endl;
         
}


int main (int argc, char** argv) {
       
    evaluate_algorithms();
    
}





//    Matrix4f transform;
//    transform <<   0.999854,   0.00993521,  0.0093811,    0.068779,
//                  -0.00991821, 0.999913,   -0.000248864, -0.0604227,
//                  -0.00940646, 0.000139605, 0.999935,     0.0793181,
//                   0.0,        0.0,         0.0,          1.0;
//    Matrix4f inv_transform = transform.inverse();
//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGBA>);
//    pcl::transformPointCloud(*cloud, *transformed, inv_transform);





