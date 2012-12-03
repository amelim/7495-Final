// Adapted from PCL tutorial on Euclidean cluster extraction and region-
// growing segmentation.

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
//#include <pcl/segmentation/region_growing.h>


pcl::PointCloud<pcl::PointXYZRGBA>::Ptr read_pcd() {
  
  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGBA>);
  
  reader.read("data/global.pcd", *cloud);
  
  std::vector<int> indices; 
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices); 
  
  return cloud;
}


void downsample(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output_cloud) {
  // Create the filtering object and downsample the dataset
  float leaf_size = 0.01f;  // 0.01 is 1 cm
  pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
  vg.setInputCloud(input_cloud);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*output_cloud);
}


std::vector<pcl::PointIndices> euclidean_clusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {

  // Create the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
  ec.setClusterTolerance(0.02); // 0.01 is 1cm
  ec.setMinClusterSize(25);
  ec.setMaxClusterSize(1000000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(clusters);

  return clusters;
}


//std::vector<pcl::PointIndices> region_growing_segmentation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {
//  
//  // Estimate surface normals
//  pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
//  pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
//  pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normal_estimator;
//  normal_estimator.setSearchMethod(tree);
//  normal_estimator.setInputCloud(cloud);
//  normal_estimator.setKSearch(50);
//  normal_estimator.compute(*normals);
//  
//  // Filter out some of the points
////  pcl::IndicesPtr indices (new std::vector <int>);
////  pcl::PassThrough<pcl::PointXYZRGBA> pass;
////  pass.setInputCloud (cloud);
////  pass.setFilterFieldName ("z");
////  pass.setFilterLimits (0.0, 1.0);
////  pass.filter (*indices);
//  
//  // Set up the region-growing segmentation stuff
//  pcl::RegionGrowing<pcl::PointXYZRGBA, pcl::Normal> reg;
//  reg.setMinClusterSize(100);
//  reg.setMaxClusterSize(10000);
//  reg.setSearchMethod(tree);
//  reg.setNumberOfNeighbours(30);
//  reg.setInputCloud(cloud);
//  //reg.setIndices (indices);
//  reg.setInputNormals(normals);
//  reg.setSmoothnessThreshold(7.0 / 180.0 * M_PI);
//  reg.setCurvatureThreshold(1.0);
//  
//  // Perform the segmentation and save the result
//  std::vector <pcl::PointIndices> clusters;
//  reg.extract(clusters);
//  
//  // Visualize the segmented point cloud
////  pcl::PointCloud <pcl::PointXYZRGBARGB>::Ptr colored_cloud = reg.getColoredCloud ();
////  pcl::visualization::CloudViewer viewer ("Cluster viewer");
////  viewer.showCloud(colored_cloud);
////  while (!viewer.wasStopped ())
////  {
////  }
//  
//  return clusters;
//}


void test_segmentation() {
  
  // Read in point cloud and downsample
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = read_pcd();
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr small_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  downsample(cloud, small_cloud);
  
  // Extract Euclidean clusters
  std::vector<pcl::PointIndices> clusters = euclidean_clusters(small_cloud);
  
  // Use region-growing segmentation
//  std::vector<pcl::PointIndices> clusters = region_growing_segmentation(small_cloud);
  
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

}






/* For now, do not use the planar segmentation stuff
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGBA> ());
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
    pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
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
