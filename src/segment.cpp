// Adapted from PCL tutorial on Euclidean cluster extraction

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


pcl::PointCloud<pcl::PointXYZ>::Ptr read_pcd() {
  
  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  
  reader.read("data/work_2/kinect_5.pcd", *cloud);
  
  return cloud;
}


void downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud) {
  // Create the filtering object and downsample the dataset
  float leaf_size = 0.005f;  // use a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(input_cloud);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*output_cloud);
}


std::vector<pcl::PointIndices> extract_clusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

  // Create the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(0.01); // 2cm
  ec.setMinClusterSize(50);
  ec.setMaxClusterSize(25000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(clusters);

  return clusters;
}

int main (int argc, char** argv) {
    
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = read_pcd();
  pcl::PointCloud<pcl::PointXYZ>::Ptr small_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  downsample(cloud, small_cloud);
  std::vector<pcl::PointIndices> clusters = segment_cloud(small_cloud);
    
  // Print out cloud and cluster sizes
  std::cout << "Points before filtering: " << cloud->points.size() << std::endl;
  std::cout << "Points after filtering: " << small_cloud->points.size() << std::endl;
  std::cout << "Number of clusters: " << clusters.size() << std::endl;
  int sum = 0;
  for (int i = 0; i < clusters.size(); i++) {
    std::cout << "Points in cluster " << i << ": " << clusters[i].indices.size() << std::endl;
    sum += clusters[i].indices.size();
  }
  std::cout << "Total points in clusters: " << sum << std::endl;
    
    return 0;
}






/* For now, do not use the planar segmentation stuff
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.02); // 2 cm

  int i = 0, nr_points = (int) cloud->points.size();
  while (cloud->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);

    // Write the planar inliers to disk
    extract.filter(*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_f);
    *cloud = *cloud_f;
  }
*/
