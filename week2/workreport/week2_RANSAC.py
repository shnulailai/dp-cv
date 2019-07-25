#!/usr/bin/env python
# coding: utf-8

# In[1]:


#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A, 
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding 
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like: 
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#       
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while 
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold 
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List


# In[1]:


def ransacMatching(A,B):
    # A & B: List of List
    '''
    pseudo code:
    
    Input: inlier ← random.sample(A,4) + random.sample(B,4)
    Output: H （homography matrix）
    Initial:outlier ← [],k ← 0,t ← 1,threhold ← 0.005（阈值）
    Procedure:
    step1：compute homography of inlier:H 
    step2：while k< 1000 and t !=0:
                 t ← 0
                 if outlier is not None:
                       for point in outlier:
                           error ← the error of points with H
                           if e < threhold: 
                                 point is new inlier
                                 t ← 1
                  if inlier is not None:
                       for point in outlier:
                           error ← the error of points with H
                           if e >= threhold: 
                                 point is new inlier
                                 t ← 1
                  sample_inlier ← random.sample(inlier,4)
                  H ← homography of sample_inlier
                  k ← k + 1
            return H
    '''


# In[3]:


def main():
    
    


# In[5]:


if __name__ == "__main":
    main()

