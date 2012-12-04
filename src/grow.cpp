#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZRGBA> ("/home/andrew/School/CV-3/data/tripod_3/kinect_1000.pcd", 
        *cloud) == -1)
  {
      std::cout << "Cloud reading failed." << std::endl;
      return (-1);
  }

  std::vector<int> idx;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);

  pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree = 
    boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA> > (new pcl::search::KdTree<pcl::PointXYZRGBA>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (100);
  normal_estimator.compute (*normals);

  std::cout << "PassThrough" << std::endl;

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZRGBA> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);

  std::cout << "Region" << std::endl;

  pcl::RegionGrowingRGB<pcl::PointXYZRGBA, pcl::Normal> reg;
  //reg.setMinClusterSize (50);
  //reg.setMaxClusterSize (10000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (50);
  reg.setInputCloud (cloud);
  reg.setPointColorThreshold(12);
  reg.setRegionColorThreshold(7);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
  std::cout << "These are the indices of the points of the initial" <<
  std::endl << "cloud that belong to the first cluster:" << std::endl;
  int counter = 0;
  while (counter < 5 || counter > clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << std::endl;
    counter++;
  }
  
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped ())
  {
  }
  return (0);
}
