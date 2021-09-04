import rospy
import os
import sys
import pcl
import numpy as np
import ctypes
import struct
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from std_msgs.msg import Header
from random import randint
from pandas import DataFrame  
from sklearn.preprocessing import StandardScaler
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import tf
class cluster_pointcloud:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pcl_sub = rospy.Subscriber("/camera/depth/points", pc2.PointCloud2, self.pcl_callback, queue_size=1)
       
         

    def ros_to_pcl(self,ros_cloud):
        points_list = []

        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])

        pcl_data = pcl.PointCloud_PointXYZRGB()
        pcl_data.from_list(points_list)

        return pcl_data

    def pcl_to_ros(self,pcl_array):
        ros_msg = PointCloud2()

        ros_msg.header.stamp = rospy.Time.now()
        ros_msg.header.frame_id = "kinect_frame"

        ros_msg.height = 1
        ros_msg.width = pcl_array.size

        ros_msg.fields.append(PointField(
                                name="x",
                                offset=0,
                                datatype=PointField.FLOAT32, count=1))
        ros_msg.fields.append(PointField(
                                name="y",
                                offset=4,
                                datatype=PointField.FLOAT32, count=1))
        ros_msg.fields.append(PointField(
                                name="z",
                                offset=8,
                                datatype=PointField.FLOAT32, count=1))
        ros_msg.fields.append(PointField(
                                name="rgb",
                                offset=16,
                                datatype=PointField.FLOAT32, count=1))

        ros_msg.is_bigendian = False
        ros_msg.point_step = 32
        ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
        ros_msg.is_dense = False
        buffer = []

        for data in pcl_array:
            s = struct.pack('>f', data[3])
            i = struct.unpack('>l', s)[0]
            pack = ctypes.c_uint32(i).value

            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)

            buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))

        ros_msg.data = "".join(buffer)

        return ros_msg
    def transform_cloud(self,cloud):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', cloud.header.frame_id,
                                                cloud.header.stamp,
                                                rospy.Duration(0.1))
        except tf2.LookupException as ex:
            rospy.logwarn(str(lookup_time.to_sec()))
            rospy.logwarn(ex)
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(str(lookup_time.to_sec()))
            rospy.logwarn(ex)
            return
        cloud_out = do_transform_cloud(cloud, trans)
        return cloud_out    

    def estimate_number_of_clusters(self,downsampled_list):
        #scaler = StandardScaler()
        downsampled_list_np=np.array(downsampled_list)
        #downsampled_list_np=scaler.fit_transform(downsampled_list_np)
    
        Data = {'x':downsampled_list_np[:,0],'y':downsampled_list_np[:,1],'z':downsampled_list_np[:,2]}
        
       
        df = DataFrame(Data,columns=['x','y','z'])    
        clustering=DBSCAN(eps=0.25,min_samples=25).fit(df)
        cluster=clustering.labels_
        number_of_clusters=len(set(cluster))
       
        Data_new= {'x':downsampled_list_np[:,0],'y':downsampled_list_np[:,1],'z':downsampled_list_np[:,2],'label':cluster}
        df_new= DataFrame(Data_new)
        return df_new,number_of_clusters
          

    
 
 
    def pcl_callback(self,data):
        cloud_new=self.ros_to_pcl(data)
        seg = cloud_new.make_segmenter()
        # Set the model you wish to fit 
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        # Max distance for a point to be considered fitting the model
        # Experiment with different values for max_distance 
        max_distance = 0.004
        seg.set_distance_threshold(max_distance)
        # Call the segment function to obtain set of inlier indices and model coefficients
        inliers, coefficients = seg.segment()
    
        extracted_ground   = cloud_new.extract(inliers, negative=False)
    
        extracted_objects = cloud_new.extract(inliers, negative=True)
        vox = extracted_objects.make_voxel_grid_filter()
        LEAF_SIZE = 0.04 #increase to filter more
        # # Set the voxel (or leaf) size
        vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
        downsampled = vox.filter()
        
        out=self.pcl_to_ros(downsampled)
        out_trans=self.transform_cloud(out)
       
        downsampled_list=[]
        for data in pc2.read_points(out_trans,field_names = ("x", "y", "z"),skip_nans=True):
            downsampled_list.append([data[1], data[2], data[0]])
        #print(downsampled_list)

        df_grouped,optimal_number_of_clusters=self.estimate_number_of_clusters(downsampled_list)
        print('Number of clusters : {}'.format(optimal_number_of_clusters))
        fig = plt.figure(figsize = (10, 7))
        ax = Axes3D(fig)
        for grp_name, grp_idx in df_grouped.groupby('label').groups.items():
            y = df_grouped.iloc[grp_idx,1]
            x = df_grouped.iloc[grp_idx,0]
            z = df_grouped.iloc[grp_idx,2]
            ax.scatter(x,y,z,label=grp_name) 
            #ax.scatter(*df_grouped.iloc[grp_idx, [0, 1, 2]].T.values, label=grp_name)(or do it in one line)
        plt.title('Pointcloud_clustering- Number of unique clusters:{}'.format(optimal_number_of_clusters))
        ax.set_ylabel('y-axis')
        ax.set_xlabel('x-axis')
        ax.set_zlabel('z-axis')      

        plt.show()  

       
    
    
def main(args):
  rospy.init_node('cluster_pointcloud', anonymous=True)
  cp = cluster_pointcloud()
  
  try:
    rospy.spin()
  except rospy.ROSInterruptException:
    pass

if __name__ == '__main__':
    main(sys.argv)

    
