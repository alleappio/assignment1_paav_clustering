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
#include <pcl/filters/crop_box.h>
#include "../include/Renderer.hpp"
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <chrono>
#include <unordered_set>
#include <pcl/filters/approximate_voxel_grid.h>
#include "../include/tree_utilities.hpp"

//#define USE_PCL_LIBRARY
using namespace lidar_obstacle_detection;

typedef std::unordered_set<int> my_visited_set_t;

/*
    * voxel size: (0.2f, 0.2f, 0.2f) 
    * inlayers dimension: 0.25;
    * floor - environment rate: 0.5
    * cluster tollerance: 0.27
    * MinClusterSize: 100
    * MaxClusterSize: 25000
*/

//This function sets up the custom kdtree using the point cloud
void setupKdtree(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, my_pcl::KdTree* tree, int dimension){
    //insert point cloud points into tree
    for (int i = 0; i < cloud->size(); ++i)
    {
        tree->insert({cloud->at(i).x, cloud->at(i).y, cloud->at(i).z}, i);
    }
}

/*
OPTIONAL
This function computes the nearest neighbors and builds the clusters
    - Input:
        + cloud: Point cloud to be explored
        + target_ndx: i-th point to visit
        + tree: kd tree for searching neighbors
        + distanceTol: Distance tolerance to build the clusters 
        + visited: Visited points --> typedef std::unordered_set<int> my_visited_set_t;
        + cluster: Here we add points that will represent the cluster
        + max: Max cluster size
    - Output:
        + visited: already visited points
        + cluster: at the end of this function we will have one cluster
*/
void proximity(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int target_ndx, my_pcl::KdTree* tree, float distanceTol, my_visited_set_t& visited, std::vector<int>& cluster, int max){
	if (cluster.size() < max){
        cluster.push_back(target_ndx);
        visited.insert(target_ndx);

        std::vector<float> point {cloud->at(target_ndx).x, cloud->at(target_ndx).y, cloud->at(target_ndx).z};
    
        // get all neighboring indices of point
        std::vector<int> neighborNdxs = tree->search(point, distanceTol);

        for (int neighborNdx : neighborNdxs){
            // if point was not visited
            if (visited.find(neighborNdx) == visited.end()){
                proximity(cloud, neighborNdx, tree, distanceTol, visited, cluster, max);
            }

            if (cluster.size() >= max){
                return;
            }
        }
    }
}

/*
OPTIONAL
This function builds the clusters following a euclidean clustering approach
    - Input:
        + cloud: Point cloud to be explored
        + tree: kd tree for searching neighbors
        + distanceTol: Distance tolerance to build the clusters 
        + setMinClusterSize: Minimum cluster size
        + setMaxClusterSize: Max cluster size
    - Output:
        + cluster: at the end of this function we will have a set of clusters
TODO: Complete the function
*/
std::vector<pcl::PointIndices> euclideanCluster(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, my_pcl::KdTree* tree, float distanceTol, int setMinClusterSize, int setMaxClusterSize){
	my_visited_set_t visited{};                                                          //already visited points
	std::vector<pcl::PointIndices> clusters;                                             //vector of PointIndices that will contain all the clusters
    std::vector<int> cluster;                                                           //vector of int that is used to store the points that the function proximity will give me back
    
    for(int i=0; i<cloud->points.size(); i++){
        if(visited.find(i) == visited.end()){
            proximity(cloud, i, tree, distanceTol, visited, cluster, setMaxClusterSize);
            if(cluster.size()>setMinClusterSize){
                pcl::PointIndices c; 
                c.indices = cluster;
                clusters.push_back(c);
            }
        }
        cluster.clear();
    }
	return clusters;	
}

void ProcessAndRenderPointCloud (Renderer& renderer, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){
    // 1) Downsample the dataset 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointXYZ source_point(0,0,0);
    std::vector<Color> colors = {Color(1,1,1), Color(0,0,1), Color(1,0,0), Color(0,1,1), Color(0,1,0)};

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.2f, 0.2f, 0.2f); 
    sor.filter(*cloud_filtered);

    // 2) here we crop the points that are far away from us, in which we are not interested
    pcl::CropBox<pcl::PointXYZ> cb(true);
    cb.setInputCloud(cloud_filtered);
    cb.setMin(Eigen::Vector4f (-20, -6, -2, 1));
    cb.setMax(Eigen::Vector4f ( 30, 7, 5, 1));
    cb.filter(*cloud_filtered); 

    // 3) Segmentation and apply RANSAC
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    int i = 0, nr_points = (int) cloud_filtered->size ();

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.25); 
    
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ()); 
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ()); 
    
    // 4) iterate over the filtered cloud, segment and remove the planar inliers 
    while (cloud_filtered->size () > 0.5 * nr_points){
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients); 
        if (inliers->indices.size () == 0){
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }
        
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered); 
        extract.setIndices (inliers);
        extract.setNegative (false); 
        extract.filter (*cloud_segmented);

        extract.setNegative (true); 
        extract.filter(*cloud_plane);
        
        cloud_filtered.swap(cloud_plane);
        i++;
    }

    // 5) Create the KDTree and the vector of PointIndices
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered); 
    
    // 6) Set the spatial tolerance for new cluster candidates (pay attention to the tolerance!!!)

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    std::vector<pcl::PointIndices> cluster_indices;


    #ifdef USE_PCL_LIBRARY
        ec.setClusterTolerance (0.27); 

        ec.setMinClusterSize (100);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered);
        
        ec.extract(cluster_indices);
        
    #else
        my_pcl::KdTree treeM;
        treeM.set_dimension(3);
        setupKdtree(cloud_filtered, &treeM,3);
        cluster_indices = euclideanCluster(cloud_filtered, &treeM, 0.27, 100, 25000);
    #endif

    /**Now we extracted the clusters out of our point cloud and saved the indices in cluster_indices. 

    To separate each cluster out of the vector<PointIndices> we have to iterate through cluster_indices, create a new PointCloud for each entry and write all points of the current cluster in the PointCloud.
    Compute euclidean distance
    **/
    int j = 0;
    int clusterId = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        cloud_cluster->push_back ((*cloud_filtered)[*pit]); 
        cloud_cluster->width = cloud_cluster->size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 7) render the cluster and plane without rendering the original cloud  
	    renderer.RenderPointCloud(cloud_segmented,"cloud segmented"+std::to_string(clusterId),colors[0]);
        renderer.RenderPointCloud(cloud_cluster,"cloud cluster"+std::to_string(clusterId),colors[1]);

        //Here we create the bounding box on the detected clusters
        pcl::PointXYZ minPt, maxPt;

        pcl::getMinMax3D(*cloud_cluster, minPt, maxPt);

        // 8) plot the distance of each cluster w.r.t ego vehicle
        Box box{minPt.x, minPt.y, minPt.z,
        maxPt.x, maxPt.y, maxPt.z};

        pcl::PointXYZ center((minPt.x + maxPt.x)/2.0f,
                                (minPt.y + maxPt.y)/2.0f,
                                (minPt.z + maxPt.z)/2.0f);

        float distance = std::sqrt(std::pow(source_point.x - center.x, 2) + 
                                    std::pow(source_point.y - center.y, 2) +
                                    std::pow(source_point.z - center.z, 2));

        renderer.addText(center.x, center.y, center.z+10, std::to_string(distance));

        // 9) Here you can color the vehicles that are both in front and 5 meters away from the ego vehicle
        //      if 5 meters away and in ftont: color "red"
        //      else: color "blue"
        
        if(distance>=5 && center.x>source_point.x){
            renderer.RenderBox(box, j, colors[2]);
        }else{
            renderer.RenderBox(box, j, colors[3]);
        }
        ++clusterId;
        j++;
    }  

}


int main(int argc, char* argv[]){
    // Read from cli dataset path
    if(argc!=2){
        std::cerr << "no cloud stream directory specified, exiting now" << std::endl;
        return 1;
    }

    Renderer renderer;
    renderer.InitCamera(CameraAngle::XY);
    // Clear viewer
    renderer.ClearViewer();

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    std::vector<boost::filesystem::path> stream(boost::filesystem::directory_iterator{argv[1]},
                                                boost::filesystem::directory_iterator{});

    // sort files in ascending (chronological) order
    std::sort(stream.begin(), stream.end());

    auto streamIterator = stream.begin();
    
    while (not renderer.WasViewerStopped()){
        renderer.ClearViewer();
        
        pcl::PCDReader reader;
        reader.read (streamIterator->string(), *input_cloud);
        auto startTime = std::chrono::steady_clock::now();

        ProcessAndRenderPointCloud(renderer,input_cloud);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        //std::cout << "[PointCloudProcessor<PointT>::ReadPcdFile] Loaded "
        //<< input_cloud->points.size() << " data points from " << streamIterator->string() <<  "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

        streamIterator++;
        if(streamIterator == stream.end())
            streamIterator = stream.begin();
        
        //renderer.RenderPointCloud(input_cloud, "pcl");

        renderer.SpinViewerOnce();
    }
}
