/*
 * Andrew Melim
 * Adapted from PCL tutorial on incremental ICP registration
 */

#include <pcl/io/pcd_io.h>

#include <pcl/point_types.h>
#include <pcl/point_representation.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/ia_ransac.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>

#include <boost/filesystem.hpp>

#include <string>
#include <vector>

using namespace std;
using namespace boost;
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//Typedefs
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

boost::filesystem::path pcd_dir("/home/andrew/School/Kinect/data/work_2");

pcl::visualization::PCLVisualizer *p;
int vp_1, vp_2;

void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud("vp1_target");
  p->removePointCloud("vp1_source");

  PointCloudColorHandlerCustom<PointType> tgt_h(cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointType> src_h(cloud_target, 255, 0, 0);

  p->addPointCloud(cloud_target, tgt_h, "vp1_target", vp_1);
  p->addPointCloud(cloud_source, src_h, "vp1_source", vp_1);
  PCL_INFO("Press q to begin registration. \n");
  p->spin();
}

void showCloudsRightFeats(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud("target");
  p->removePointCloud("source");

  PointCloudColorHandlerCustom<PointType> tgt_h(cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointType> src_h(cloud_target, 255, 0, 0);

  p->addPointCloud(cloud_target, tgt_h, "target", vp_2);
  p->addPointCloud(cloud_source, src_h, "source", vp_2);
  p->spinOnce();
}

void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target,
    const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud("source");
  p->removePointCloud("target");

  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler(cloud_target, "curvature");
  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler(cloud_source, "curvature");

  p->addPointCloud(cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud(cloud_source, src_color_handler, "source", vp_2);

  p->spinOnce();
}

class PointRep : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
  public:
    PointRep()
    {
      nr_dimensions_ = 4;
    }

    virtual void copyToFloatArray(const PointNormalT &p, float * out) const
    {
      // <x,y,z,curvature>
      out[0] = p.x;
      out[1] = p.y;
      out[2] = p.z;
      out[3] = p.curvature;
    }
};

void pairAlign(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, 
    PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  PointCloud::Ptr src(new PointCloud);
  PointCloud::Ptr tgt(new PointCloud);
  pcl::VoxelGrid<PointType> grid;
  if(downsample)
  {
    grid.setLeafSize(0.05, 0.05, 0.05);
    grid.setInputCloud(cloud_src);
    grid.filter(*src);

    grid.setInputCloud(cloud_tgt);
    grid.filter(*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }

  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointType, PointNormalT> norm_est;
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType> ());
  norm_est.setSearchMethod(tree);
  norm_est.setKSearch(30);

  norm_est.setInputCloud(src);
  norm_est.compute(*points_with_normals_src);
  pcl::copyPointCloud(*src, *points_with_normals_src);

  norm_est.setInputCloud(tgt);
  norm_est.compute(*points_with_normals_tgt);
  pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

  PointRep point_representation;
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues(alpha);

  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon(1e-5);
  reg.setMaxCorrespondenceDistance(0.15);
  reg.setPointRepresentation(boost::make_shared<const PointRep> (point_representation));

  reg.setInputCloud(points_with_normals_src);
  reg.setInputTarget(points_with_normals_tgt);

  //Optimize
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations(2);

  for(int i = 0; i < 300; ++i)
  {
    points_with_normals_src = reg_result;

    reg.setInputCloud(points_with_normals_src);
    reg.align(*reg_result);

    Ti = reg.getFinalTransformation() * Ti;

    if(fabs ((reg.getLastIncrementalTransformation() - prev).sum()) < 
        reg.getTransformationEpsilon())
      reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

    prev = reg.getLastIncrementalTransformation();

    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
    std::cout << "Fitness: " << reg.getFitnessScore() << std::endl;
  }

  targetToSource = Ti.inverse();

  pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);
  p->removePointCloud ("source");
  p->removePointCloud ("target");

  PointCloudColorHandlerCustom<PointType> cloud_tgt_h (output, 0, 255, 0);
  PointCloudColorHandlerCustom<PointType> cloud_src_h (cloud_src, 255, 0, 0);
  p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

  PCL_INFO ("Press q to continue the registration.\n");
  p->spin ();

  *output += *cloud_src;
  final_transform = targetToSource;
}

//Problem with ICP
//Gets stuck in local minima with outliers
//Needs initial alignment
void featureAlign(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, 
    PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  PointCloud::Ptr src(new PointCloud);
  PointCloud::Ptr tgt(new PointCloud);
  pcl::VoxelGrid<PointType> grid;
  if(downsample)
  {
    grid.setLeafSize(0.03, 0.03, 0.03);
    grid.setInputCloud(cloud_src);
    grid.filter(*src);

    grid.setInputCloud(cloud_tgt);
    grid.filter(*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }

  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointType, PointNormalT> norm_est;
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType> ());
  norm_est.setSearchMethod(tree);
  //norm_est.setKSearch(30);
  norm_est.setRadiusSearch(0.05);

  norm_est.setInputCloud(src);
  norm_est.compute(*points_with_normals_src);
  pcl::copyPointCloud(*src, *points_with_normals_src);

  norm_est.setInputCloud(tgt);
  norm_est.compute(*points_with_normals_tgt);
  pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

  pcl::PFHEstimation<PointType, PointNormalT, pcl::PFHSignature125> pfh;
  pcl::search::KdTree<PointType>::Ptr pfh_tree(new pcl::search::KdTree<PointType>());

  pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs_src (new pcl::PointCloud<pcl::PFHSignature125> ());
  pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs_tgt (new pcl::PointCloud<pcl::PFHSignature125> ());

  pfh.setRadiusSearch(0.5);
  pfh.setSearchMethod(tree);

  pfh.setInputCloud(src);
  pfh.setInputNormals(points_with_normals_src);
  pfh.compute(*pfhs_src);

  pfh.setInputCloud(tgt);
  pfh.setInputNormals(points_with_normals_tgt);
  pfh.compute(*pfhs_tgt);

  //Initial Alignment with SACIA

  pcl::PointCloud<PointType>::Ptr ia_cloud(new pcl::PointCloud<PointType>);
  pcl::SampleConsensusInitialAlignment<PointType, PointType, pcl::PFHSignature125> sac;

  sac.setMinSampleDistance(0.05f);
  sac.setMaxCorrespondenceDistance(0.5);
  sac.setMaximumIterations(500);

  sac.setInputCloud(src);
  sac.setSourceFeatures(pfhs_src);

  sac.setInputTarget(tgt);
  sac.setTargetFeatures(pfhs_tgt);

  sac.align(*ia_cloud);
  Eigen::Matrix4f ia_trans = sac.getFinalTransformation();
  showCloudsRightFeats(ia_cloud, tgt);

  //Refinement
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(0.5);
  icp.setRANSACOutlierRejectionThreshold(0.1);
  icp.setTransformationEpsilon(1e-6);
  icp.setMaximumIterations(200);

  icp.setInputCloud(ia_cloud);
  icp.setInputTarget(tgt);

  icp.align(*output);

  final_transform = icp.getFinalTransformation() * ia_trans;


  showCloudsRightFeats(output, tgt);
  PCL_INFO ("Press q to continue the registration.\n");
  p->spin ();

}

int main(int argc, char** argv)
{
  
  pcl::PointCloud<PointType>::Ptr src_cloud(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr tgt_cloud(new pcl::PointCloud<PointType>);
  PointCloud::Ptr result(new PointCloud);
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;

  //Sort of input values
  vector<boost::filesystem::path> pcd_files;
  vector<boost::filesystem::path>::iterator pcd_itr;
  copy(boost::filesystem::directory_iterator(pcd_dir),
      boost::filesystem::directory_iterator(), back_inserter(pcd_files));
  sort(pcd_files.begin(), pcd_files.end());

  //Init PCLVisualizer
  p = new pcl::visualization::PCLVisualizer(argc, argv, "Pairwise Incremental ICP");
  p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
  p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);

  for(int j = 1; j < pcd_files.size(); ++j)
  {
    boost::filesystem::path source = pcd_files[j-1];
    boost::filesystem::path target = pcd_files[j];
    if(pcl::io::loadPCDFile<PointType>(source.string(), *src_cloud) == -1)
    {
      PCL_ERROR("Couldn't read source file!");
    }
    if(pcl::io::loadPCDFile<PointType>(target.string(), *tgt_cloud) == -1)
    {
      PCL_ERROR("Couldn't read source file!");
    }
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*src_cloud, *src_cloud, indices);

    showCloudsLeft(src_cloud, tgt_cloud);

    PointCloud::Ptr temp(new PointCloud);
    //pairAlign(src_cloud, tgt_cloud, temp, pairTransform, true);
    featureAlign(src_cloud, tgt_cloud, temp, pairTransform, true);

    pcl::transformPointCloud(*temp, *result, GlobalTransform);
    GlobalTransform = pairTransform * GlobalTransform;

  }

  return 0;
}
