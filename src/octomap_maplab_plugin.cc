#include <iostream>
#include <string>
#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <Eigen/Core>
#include <aslam/common/pose-types.h>
#include <console-common/console-plugin-base.h>
#include <map-manager/map-manager.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <landmark-triangulation/pose-interpolator.h>
#include <map-resources/resource-conversion.h>
#include <maplab-common/progress-bar.h>
#include <posegraph/unique-id.h>
#include <vi-map/landmark.h>
#include <vi-map/unique-id.h>
#include <vi-map/vertex.h>
#include <vi-map/vi-map.h>
#include <voxblox/core/common.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>

#include <octomap/octomap.h>
#include <opencv2/highgui.hpp>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <rosbag/bag.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/TransformStamped.h>
#include <minkindr_conversions/kindr_msg.h>
// Your new plugin needs to derive from ConsolePluginBase.
// (Alternatively, you can derive from ConsolePluginBaseWithPlotter if you need
// RViz plotting abilities for your VI map.)
class OctoMapPlugin : public common::ConsolePluginBase
{
public:
  // Every plugin needs to implement a getPluginId function which returns a
  // string that gives each plugin a unique name.
  std::string getPluginId() const override
  {
    return "octomap_maplab_plugin";
  }

  // The constructor takes a pointer to the Console object which we can forward
  // to the constructor of ConsolePluginBase.
  OctoMapPlugin(common::Console *console)
      : common::ConsolePluginBase(console)
  {
    // You can add your commands in here.
    addCommand(
        {"hello_world", "hello"}, // Map "hello_world" and "hello" to this
                                  // command.
        [this]() -> int {         // Function to call when this command is entered.
          // This function can do anything you want. Check the other plugins
          // under ~/maplab_ws/src/maplab/console-plugins for more examples.

          // Here, we just print a message to the terminal.
          std::cout << "Hello world!" << std::endl;

          // Every console command returns an integer, you can take one from
          // the CommandStatus enum. kSuccess returns everything is fine.
          // Other commonly used return values are common::kUnknownError and
          // common::kStupidUserError.
          return common::kSuccess;
        },

        // This is the description of your command. This will get printed when
        // you run `help` in the console.
        "This command prints \"Hello World!\" to the console.",

        // This specifies the execution method of your command. For most
        // commands, it is sufficient to run them in sync with
        // common::Processing::Sync.
        common::Processing::Sync);

    addCommand(
        {"create_octomap"},

        [this]() -> int {
          // Get the currently selected map.
          std::string selected_map_key;

          // This function will write the name of the selected map key into
          // selected_map_key. The function will return false and print an error
          // message if no map key is selected.
          if (!getSelectedMapKeyIfSet(&selected_map_key))
          {
            return common::kStupidUserError;
          }

          // Create a map manager instance.
          vi_map::VIMapManager map_manager;

          // Get and lock the map which blocks all other access to the map.
          vi_map::VIMapManager::MapWriteAccess map =
              map_manager.getMapWriteAccess(selected_map_key);
          const vi_map::VIMap &vi_map = *map;

          // get mission lists
          vi_map::MissionIdList mission_ids;
          map.get()->getAllMissionIdsSortedByTimestamp(&mission_ids);

          // set resource type 17 pointcloudxyz
          const backend::ResourceType input_resource_type = static_cast<backend::ResourceType>(
              17);

          // octomap tree
          octomap::OcTree tree(0.05);

          // pcl point cloud
          typedef pcl::PointXYZ PointT;
          typedef pcl::PointCloud<PointT> PointCloud;
          PointCloud::Ptr pointCloud(new PointCloud);

          // rosbag
          rosbag::Bag bag;
          bag.open("/home/nrslnuc2/Dataset/pointcloud_processed.bag", rosbag::bagmode::Write);

          for (const vi_map::MissionId &mission_id : mission_ids)
          {
            VLOG(1) << "Integrating mission " << mission_id;
            const vi_map::VIMission &mission = vi_map.getMission(mission_id);

            const aslam::Transformation &T_G_M =
                vi_map.getMissionBaseFrameForMission(mission_id).get_T_G_M();

            // Check if there is IMU data to interpolate the optional sensor poses.
            landmark_triangulation::VertexToTimeStampMap vertex_to_time_map;
            int64_t min_timestamp_ns;
            int64_t max_timestamp_ns;
            const landmark_triangulation::PoseInterpolator pose_interpolator;
            pose_interpolator.getVertexToTimeStampMap(
                vi_map, mission_id, &vertex_to_time_map, &min_timestamp_ns,
                &max_timestamp_ns);
            if (vertex_to_time_map.empty())
            {
              VLOG(2) << "Couldn't find any IMU data to interpolate exact optional "
                      << "sensor position in mission " << mission_id;
              continue;
            }

            LOG(INFO) << "All resources within this time range will be integrated: ["
                      << min_timestamp_ns << "," << max_timestamp_ns << "]";

            // Retrieve sensor id to resource id mapping.
            typedef std::unordered_map<aslam::CameraId,
                                       backend::OptionalSensorResources>
                SensorsToResourceMap;

            const SensorsToResourceMap *sensor_id_to_res_id_map;
            sensor_id_to_res_id_map =
                mission.getAllOptionalSensorResourceIdsOfType<aslam::CameraId>(
                    input_resource_type);

            if (sensor_id_to_res_id_map == nullptr)
            {
              continue;
            }
            VLOG(1) << "Found " << sensor_id_to_res_id_map->size()
                    << " optional sensors with this depth type.";

            // Integrate them one sensor at a time.
            for (const typename SensorsToResourceMap::value_type &sensor_to_res_ids :
                 *sensor_id_to_res_id_map)
            {
              const backend::OptionalSensorResources &resource_buffer =
                  sensor_to_res_ids.second;

              const aslam::CameraId &sensor_or_camera_id = sensor_to_res_ids.first;

              // Get transformation between reference (e.g. IMU) and sensor.
              aslam::Transformation T_I_S;
              vi_map.getSensorManager().getSensorOrCamera_T_R_S(
                  sensor_or_camera_id, &T_I_S);

              const size_t num_resources = resource_buffer.size();
              VLOG(1) << "Sensor " << sensor_or_camera_id.shortHex() << " has "
                      << num_resources << " such resources.";

              // Collect all timestamps that need to be interpolated.
              Eigen::Matrix<int64_t, 1, Eigen::Dynamic> resource_timestamps(
                  num_resources);
              size_t idx = 0u;
              for (const std::pair<int64_t, backend::ResourceId> &stamped_resource_id :
                   resource_buffer)
              {
                // If the resource timestamp does not lie within the min and max
                // timestamp of the vertices, we cannot interpolate the position. To
                // keep this efficient, we simply replace timestamps outside the range
                // with the min or max. Since their transformation will not be used
                // later, that's fine.
                resource_timestamps[idx] = std::max(
                    min_timestamp_ns,
                    std::min(max_timestamp_ns, stamped_resource_id.first));

                ++idx;
              }

              // Interpolate poses at resource timestamp.
              aslam::TransformationVector poses_M_I;
              pose_interpolator.getPosesAtTime(
                  vi_map, mission_id, resource_timestamps, &poses_M_I);

              CHECK_EQ(static_cast<int>(poses_M_I.size()), resource_timestamps.size());
              CHECK_EQ(poses_M_I.size(), resource_buffer.size());

              // Retrieve and integrate all resources.
              idx = 0u;
              int count_processed = 0;
              for (const std::pair<int64_t, backend::ResourceId> &stamped_resource_id :
                   resource_buffer)
              {

                /*
                while(1)
                {
                  int k = cv::waitKey(100);
                  if ( k == 110)
                  {
                    std::cout << "process next point cloud" << std::endl;
                    break;
                  } 
                  else if ( k == -1 )
                    continue;
                  else
                    std::cout << " k = " << k << std::endl;
                }
                */

                // We assume the frame of reference for the sensor system is the IMU
                // frame.
                const aslam::Transformation &T_M_I = poses_M_I[idx];

                // !!!!!! T_I_S actually is T_S_I
                const aslam::Transformation T_G_S = T_G_M * T_M_I * T_I_S.inverse();
                
                ++idx;

                const int64_t timestamp_ns = stamped_resource_id.first;
                // If the resource timestamp does not lie within the min and max
                // timestamp of the vertices, we cannot interpolate the position.
                if (timestamp_ns < min_timestamp_ns ||
                    timestamp_ns > max_timestamp_ns)
                {
                  LOG(WARNING) << "The optional depth resource at " << timestamp_ns
                               << " is outside of the time range of the pose graph, "
                               << "skipping.";
                  continue;
                }

                // Check if a point cloud is available.
                resources::PointCloud point_cloud;
                if (!vi_map.getOptionalSensorResource(
                        mission, input_resource_type, sensor_or_camera_id,
                        timestamp_ns, &point_cloud))
                {
                  LOG(FATAL) << "Cannot retrieve optional point cloud resources at "
                             << "timestamp " << timestamp_ns << "!";
                }

                VLOG(3) << "Found point cloud at timestamp " << timestamp_ns;

                // generate octomap
                octomap::Pointcloud cloud;

                // generate pcd file
                PointCloud::Ptr current(new PointCloud);

                // publish point cloud to rviz
                ros::NodeHandle nh;
	              ros::Publisher point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>( "/point_cloud", 10);
                sensor_msgs::PointCloud2Ptr pc = boost::make_shared<sensor_msgs::PointCloud2>();

                pc->header.frame_id = "/map";
                pc->header.stamp = ros::Time::now();
                pc->width = point_cloud.size();
	              pc->height = 1;
                pc->is_bigendian = false;
                pc->is_dense = false;

                sensor_msgs::PointCloud2Modifier pc_modifier(*pc);
                pc_modifier.setPointCloud2FieldsByString(1, "xyz");

                sensor_msgs::PointCloud2Iterator<float> iter_x(*pc, "x");
                sensor_msgs::PointCloud2Iterator<float> iter_y(*pc, "y");
                sensor_msgs::PointCloud2Iterator<float> iter_z(*pc, "z");

                Eigen::Vector3d point;
                Eigen::Vector3d pointWorld;
                Eigen::Isometry3d T(T_G_S.getTransformationMatrix());
                for (size_t index = 0u; index < point_cloud.size(); index++)
                {
                  point[0] = point_cloud.xyz[0 + 3 * index];
                  point[1] = point_cloud.xyz[1 + 3 * index];
                  point[2] = point_cloud.xyz[2 + 3 * index];

                  pointWorld = T * point;

                  cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
                  
                  // ros point cloud
                  *iter_x = pointWorld[0];
                  *iter_y = pointWorld[1];
                  *iter_z = pointWorld[2];
    
                  //*iter_x = point[0];
                  //*iter_y = point[1];
                  //*iter_z = point[2];
                  ++iter_x, ++iter_y, ++iter_z;
                }

                // save point cloud
                bag.write("/point_cloud", pc->header.stamp, *pc);

                // save the transform imu->map
                geometry_msgs::TransformStamped msg_stamped;
                geometry_msgs::Transform msg;
                tf::transformKindrToMsg(T_G_M * T_M_I, &msg);
                msg_stamped.child_frame_id = "imu";
                msg_stamped.header.stamp = pc->header.stamp;
                msg_stamped.header.frame_id = "map";
                msg_stamped.transform = msg;
                bag.write("/sensor_pose", pc->header.stamp, msg_stamped);
                
                // octomap
                tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
                count_processed++;

                LOG(INFO) << "processed point cloud " << count_processed
                          << " The total number: " << resource_buffer.size();
              }
            }
          }

          // write octomap to disk
          tree.updateInnerOccupancy();
          std::cout << "save octomap to octomap.bt!" << std::endl;
          tree.writeBinary("octomap.bt");

          return common::kSuccess;
        },

        "This command will create a octomap from point cloud data.",
        common::Processing::Sync);
  }
};

// Finally, call the MAPLAB_CREATE_CONSOLE_PLUGIN macro to create your console
// plugin.
MAPLAB_CREATE_CONSOLE_PLUGIN(OctoMapPlugin);
