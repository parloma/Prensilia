#define N_LABELS (22)

#define TARGET_DEPTH (500)

#define MAX_RESPONSE (10000)

__constant unsigned char LABEL2RGB[][3] = {{0, 0, 255},      /* thumb 3 */
					   {51, 51, 204},    /* thumb 2 */
					   {102, 102, 204},  /* thumb 1 */
					   {0, 255, 255},    /* pinky 3 */
					   {51, 255, 255},   /* pinky 2 */
					   {102, 255, 255},  /* pinky 1 */
					   {0, 255, 0},      /* ring 3 */
					   {51, 255, 51},    /* ring 2 */
					   {102, 255, 102},  /* ring 1 */
					   {255, 0, 0},      /* middle 3 */
					   {255, 51, 51},    /* middle 2 */
					   {255, 102, 102},  /* middle 1 */
					   {255, 0, 255},    /* index 3 */
					   {255, 51, 255},   /* index 2 */
					   {255, 102, 255},  /* index 1 */
					   {25, 25, 255},    /* thumb palm */
					   {153, 255, 255},  /* pinky palm */
					   {153, 255, 153},  /* ring palm */
					   {255, 153, 153},  /* middle palm */
					   {255, 153, 255},  /* index palm */
					   {0, 76, 255},     /* palm */
					   {0, 25, 76}};     /* wrist */


typedef struct TreeNode {
  int leftChild;
  short4 feature;
  short threshold;
  float histogram[N_LABELS];
} TreeNode;


typedef enum
{
  LABEL = 1,
  RGB_LABEL = 1<<1,
  POSTERIOR = 1<<2,
  HISTOGRAM = 1<<3
} mode;

signed short computeResponse(__global unsigned short *depthmap,
			     unsigned int dWidth, unsigned int dHeight,
			     short4 feature,
			     unsigned int idx)
{
  int4 offsetUV;
  float depth;
  int u, v;
  signed short response, tmp;

  // Get sample coordinates and depth value
  u = idx%dWidth;
  v = idx/dWidth;
  depth = (float)depthmap[idx];

  // Compute offsets pair coordinates
  // TODO: round instead of cast?
  offsetUV.x = u+(int)round(((float)feature.x * TARGET_DEPTH)/depth);
  offsetUV.y = v+(int)round(((float)feature.y * TARGET_DEPTH)/depth);
  offsetUV.z = u+(int)round(((float)feature.z * TARGET_DEPTH)/depth);
  offsetUV.w = v+(int)round(((float)feature.w * TARGET_DEPTH)/depth);      


  // Compute feature response
  //response = (offsetUV.x<0 || offsetUV.x>=dWidth || offsetUV.y<0 || offsetUV.y>=dHeight) ? \
      //  MAX_RESPONSE : depthmap[offsetUV.y*dWidth+offsetUV.x];
  //response -= (offsetUV.z<0 || offsetUV.z>=dWidth || offsetUV.w<0 || offsetUV.w>=dHeight) ? \
      //  MAX_RESPONSE : depthmap[offsetUV.w*dWidth+offsetUV.z];
  tmp = (offsetUV.x<0 || offsetUV.x>=dWidth || offsetUV.y<0 || offsetUV.y>=dHeight) ? \
    MAX_RESPONSE : depthmap[offsetUV.y*dWidth+offsetUV.x];
  response = (tmp) ? tmp : MAX_RESPONSE;
  tmp = (offsetUV.z<0 || offsetUV.z>=dWidth || offsetUV.w<0 || offsetUV.w>=dHeight) ? \
    MAX_RESPONSE : depthmap[offsetUV.w*dWidth+offsetUV.z];
  response -= (tmp) ? tmp : MAX_RESPONSE;

  return response;
}



__kernel void predict(__global unsigned short *depthmap,
		      unsigned int dWidth, unsigned int dHeight,
		      __global unsigned int *pixels,
		      __global unsigned char *labels,
		      __global float *posteriors,
		      unsigned int pixelsXDmap,
		      short int scale,
		      __global TreeNode *tree0,
		      __global TreeNode *tree1,
		      __global TreeNode *tree2,
		      __global TreeNode *tree3,
		      __global TreeNode *tree4,
		      //__global TreeNode *tree5,
		      //__global TreeNode *tree6,
		      //__global TreeNode *tree7,
		      //__global TreeNode *tree8,
		      //__global TreeNode *tree9,
		      unsigned int forestSize,
		      __local unsigned int *pixelLeaves,
		      int mode)

{
  unsigned int idx, currNode=0;
  signed short response;
  short4 feature;
  __global TreeNode *leaf;

  idx = pixels[get_global_id(0)];


  // ************ PER-TREE PREDICTION UNROLLED *************** //

  // Tree 0
  leaf = tree0;
  while (leaf->leftChild != -1)
  {
    feature = leaf->feature;
    feature *= scale;

    response = computeResponse(depthmap, dWidth, dHeight, feature, idx);

    currNode = (unsigned int)leaf->leftChild;
    currNode += (response<=leaf->threshold) ? 0 : 1;

    leaf = tree0+currNode;
  }
  pixelLeaves[get_local_id(0)] = currNode;
  
  // Tree 1
  if (forestSize > 1)
  {
    leaf = tree1;
    while (leaf->leftChild != -1)
    {
      feature = leaf->feature;
      feature *= scale;

      response = computeResponse(depthmap, dWidth, dHeight, feature, idx);

      currNode = (unsigned int)leaf->leftChild;
      currNode += (response<=leaf->threshold) ? 0 : 1;
      
      leaf = tree1+currNode;
    }
    pixelLeaves[get_local_id(0)+get_local_size(0)] = currNode;
  }

  // Tree 2
  if (forestSize > 2)
  {
    leaf = tree2;
    while (leaf->leftChild != -1)
    {
      feature = leaf->feature;
      feature *= scale;

      response = computeResponse(depthmap, dWidth, dHeight, feature, idx);

      currNode = (unsigned int)leaf->leftChild;
      currNode += (response<=leaf->threshold) ? 0 : 1;
      
      leaf = tree2+currNode;
    }
    pixelLeaves[get_local_id(0)+get_local_size(0)*2] = currNode;
  }

  // Tree 3
  if (forestSize > 3)
  {
    leaf = tree3;
    while (leaf->leftChild != -1)
    {
      feature = leaf->feature;
      feature *= scale;

      response = computeResponse(depthmap, dWidth, dHeight, feature, idx);

      currNode = (unsigned int)leaf->leftChild;
      currNode += (response<=leaf->threshold) ? 0 : 1;
      
      leaf = tree3+currNode;
    }
    pixelLeaves[get_local_id(0)+get_local_size(0)*3] = currNode;
  }

  // Tree 4
  if (forestSize > 4)
  {
    leaf = tree4;
    while (leaf->leftChild != -1)
    {
      feature = leaf->feature;
      feature *= scale;

      response = computeResponse(depthmap, dWidth, dHeight, feature, idx);

      currNode = (unsigned int)leaf->leftChild;
      currNode += (response<=leaf->threshold) ? 0 : 1;
      
      leaf = tree4+currNode;
    }
    pixelLeaves[get_local_id(0)+get_local_size(0)*4] = currNode;
  }

  /********* DONE WITH PER-TREE PREDICTION *********/

  // Average the posteriors of each leaf node reached and get the class
  // associated to the best one.
  float maxP = 0.0f;
  response = 0; // Will store the choosen label (avoid to declare another variable)

  for (int l=0; l<N_LABELS; l++)
  {
    float tmpP = 0.0f;

    leaf = tree0+pixelLeaves[get_local_id(0)];
    tmpP += leaf->histogram[l];

    if (forestSize>1)
    {
      leaf = tree1+pixelLeaves[get_local_id(0)+get_local_size(0)];
      tmpP += leaf->histogram[l];
    }

    if (forestSize>2)
    {
      leaf = tree2+pixelLeaves[get_local_id(0)+get_local_size(0)*2];
      tmpP += leaf->histogram[l];
    }

    if (forestSize>3)
    {      
      leaf = tree3+pixelLeaves[get_local_id(0)+get_local_size(0)*3];
      tmpP += leaf->histogram[l];
    }

    if (forestSize>4)
    {
      leaf = tree4+pixelLeaves[get_local_id(0)+get_local_size(0)*4];
      tmpP += leaf->histogram[l];
    }

    response = (tmpP>maxP) ? l : response;
    maxP = (tmpP>maxP) ? tmpP : maxP;
 
    if (mode & HISTOGRAM)
    {
      //posteriors[l*get_global_size(0)+get_global_id(0)] = tmpP/forestSize;
      posteriors[l*dWidth*dHeight+idx] = tmpP/forestSize;
    }
  }

  if (mode & RGB_LABEL)
  {
    // Save the rgb color associated to the max-probability label
    labels[idx*3] = LABEL2RGB[response][0];
    labels[idx*3+1] = LABEL2RGB[response][1];
    labels[idx*3+2] = LABEL2RGB[response][2];
  }
  else if (mode & LABEL)
  {
    labels[get_global_id(0)] = response;
  }

  if ((mode & POSTERIOR) && !(mode & HISTOGRAM))
  {
    //posteriors[get_global_id(0)] = maxP;
  }
}

#define N_BANKS (32)
#define LOG_NUM_BANKS (8)
#define CONFLICT_FREE_OFFSET(n) \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__kernel void posteriorThrNaive(__global float *posteriors,
				__global unsigned int *pixels,
				__global float *thresholds,
				unsigned int nPixels,
				unsigned int depthmapWidth,
				unsigned int depthmapHeight,
				__global unsigned int *perBlockLabPixelsCount,
				__global unsigned int *perBlockLabPixels,
				__local unsigned int *temp)
{
  unsigned int label, offset;
  bool marked1, marked2;
  unsigned int ai, bi;
  float t;

  label = get_global_id(0)/(nPixels/2);


  // ******* Prefix sum *******
  // Based on implementation by Harris07
  // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
  // TODO: handle shared memory banks access conflicts

  // Each thread work for two adjacent pixels of a different label
  ai = 2*(get_global_id(0)%(nPixels/2));
  bi = ai+1;
  
  offset = depthmapWidth*depthmapHeight*label+pixels[ai];
  marked1 = (posteriors[offset]>thresholds[label]) ? true: false;

  offset = depthmapWidth*depthmapHeight*label+pixels[bi];
  marked2 = (posteriors[offset]>thresholds[label]) ? true: false;
  
  ai = 2*get_local_id(0);
  bi = ai+1;
  temp[ai] = (marked1) ? 1 : 0;
  temp[bi] = (marked2) ? 1 : 0;

  offset = 1;
  for (int d=get_local_size(0); d>0; d>>=1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0)<d)
    {
      ai = offset*(2*get_local_id(0)+1)-1;
      bi = offset*(2*get_local_id(0)+2)-1;

      temp[bi] += temp[ai];
    }

    offset *= 2;
  }

  
  if (get_local_id(0)==0)
  {
    perBlockLabPixelsCount[get_global_id(0)/get_local_size(0)] = \
      temp[get_local_size(0)*2-1];
    temp[get_local_size(0)*2-1] = 0;
  }
  
  
  for (int d=1; d<(get_local_size(0)<<1); d*=2)
  {
    offset >>=1;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (get_local_id(0) < d)
    {
      ai = offset*(2*get_local_id(0)+1)-1;
      bi = offset*(2*get_local_id(0)+2)-1;

      t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  if (marked1)
  {
    ai = 2*(get_global_id(0)%(nPixels/2));
    offset = (get_global_id(0)/get_local_size(0))*(2*get_local_size(0))+temp[2*get_local_id(0)];
    perBlockLabPixels[offset] = pixels[ai];
  }
  if (marked2)
  {
    bi = 2*(get_global_id(0)%(nPixels/2))+1;
    offset = (get_global_id(0)/get_local_size(0))*(2*get_local_size(0))+temp[2*get_local_id(0)+1];
    perBlockLabPixels[offset] = pixels[bi];
  }
}


__kernel void compactBlockPixels(unsigned int depthmapWidth,
				 unsigned int depthmapHeight,
				 unsigned int nPixels,
				 __global unsigned int *perBlockLabPixelsCount,
				 __global unsigned int *perBlockLabPixels,
				 __global unsigned int *labelPixels,
				 __global unsigned int *perLabelPixelsCount)
{
  unsigned int label, labelBlockId, nBlocksPerLabel, nPixelsPerBlock;
  __local unsigned int offset[1];
  

  label = get_global_id(0)/(nPixels/2);
  nBlocksPerLabel = nPixels/(2*get_local_size(0));
  labelBlockId = (get_global_id(0)/get_local_size(0))%nBlocksPerLabel;
  

  // TODO: parallelize???
  if (!get_local_id(0))
  {
    *offset=0;
    for (int i=0; i<labelBlockId; i++)
    {
      *offset += perBlockLabPixelsCount[label*nBlocksPerLabel+i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);


  if (!get_local_id(0) && labelBlockId==(nBlocksPerLabel-1))
  {
    perLabelPixelsCount[label] =					\
      *offset+perBlockLabPixelsCount[label*nBlocksPerLabel+labelBlockId];
  }

  
  nPixelsPerBlock = perBlockLabPixelsCount[label*nBlocksPerLabel+labelBlockId];
  if ((get_local_id(0)*2)<nPixelsPerBlock)
  {
    labelPixels[depthmapWidth*depthmapHeight*label+*offset+get_local_id(0)*2] = perBlockLabPixels[get_global_id(0)*2];
  }
  if ((get_local_id(0)*2+1)<nPixelsPerBlock)
  {
    labelPixels[depthmapWidth*depthmapHeight*label+*offset+get_local_id(0)*2+1] = perBlockLabPixels[get_global_id(0)*2+1];
  }
}



/****************** MEAN SHIFT **********************/
#define TAN_HFOV (0.555733323f)
#define TAN_VFOV (0.41679999f)
//#define TAN_HFOV (0.5866965152984336f)
//#define TAN_VFOV (0.4224165382876819f)


#define MEANSHIFT_NITER (10)
#define MEANSHIFT_THRESHOLD (0.1f)


__kernel void updatePerPixelVotes(__global unsigned short *depthmap,
				  unsigned int depthmapWidth,
				  unsigned int depthmapHeight,
				  __global float *posteriors,
				  __global float *bandwidths,
				  __global unsigned int *pixels,
				  __global unsigned int *perLabelPixelsCount,
				  __global float3 *votes,
				  __global float *weights,
				  __local float *perBlockPoints,
				  __local float *perBlockPosteriors)
{
  float3 currPoint, m;
  float3 numTermSum = (float3)(0.0f, 0.0f, 0.0f);
  float denTermSum=0.0f;
  float w, expTerm;
  __local unsigned int temp[N_LABELS];
  
  unsigned int label=0;
  unsigned int index=0;
  unsigned int offset=0;
  unsigned int nPixels;
  bool active, toWrite;

  // Find label and output offset
  if (get_local_id(0)<N_LABELS)
  {
    temp[get_local_id(0)] = perLabelPixelsCount[get_local_id(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (label=0; label<N_LABELS; label++)
  {
    offset += temp[label];
    offset += get_local_size(0)-(temp[label]%get_local_size(0));
    if (offset>get_global_id(0))
    {
      offset -= temp[label];
      offset -= get_local_size(0)-(temp[label]%get_local_size(0));
      break;
    }
  }
  
  //index = pixels[get_global_id(0)];
  index = pixels[depthmapWidth*depthmapHeight*label+get_global_id(0)-offset];
  active = (get_global_id(0)-offset)<temp[label];
  toWrite = active;
  nPixels = temp[label];

  
  if (active)
  {
    currPoint.z = (float)depthmap[index];
    currPoint.x = 2.0f*((float)(index%depthmapWidth)/depthmapWidth-0.5f)*currPoint.z*TAN_HFOV;
    currPoint.y = 2.0f*((float)(depthmapHeight-(index/depthmapWidth))/depthmapHeight-0.5f)*currPoint.z*TAN_VFOV;
    currPoint.z *= -1.0f;
    //weights[get_global_id(0)] =					\
    //  posteriors[depthmapWidth*depthmapHeight*label+index]*currPoint.z*currPoint.z;
    weights[get_global_id(0)] =	posteriors[depthmapWidth*depthmapHeight*label+index];
  }

  
  for (int k=0; k<MEANSHIFT_NITER; k++)
  {

    for (unsigned int i=0; (get_local_size(0)*i)<nPixels; i++)
    {

      if (((get_local_size(0)*i)+get_local_id(0))<nPixels)
      {
	index = pixels[depthmapWidth*depthmapHeight*label+get_local_size(0)*i+get_local_id(0)];

	// Parallel reprojection of pixels
	perBlockPoints[get_local_id(0)] = (float)depthmap[index];
	perBlockPoints[get_local_id(0)+get_local_size(0)] =		\
	  2.0f*((float)(index%depthmapWidth)/depthmapWidth-0.5f)*perBlockPoints[get_local_id(0)]*TAN_HFOV;
	perBlockPoints[get_local_id(0)+get_local_size(0)*2] =		\
	  2.0f*((float)(depthmapHeight-(index/depthmapWidth))/depthmapHeight-0.5f)*perBlockPoints[get_local_id(0)]*TAN_VFOV;
	perBlockPoints[get_local_id(0)] *= -1.0f;

	// Parallel copy of posteriors
	perBlockPosteriors[get_local_id(0)] = posteriors[depthmapWidth*depthmapHeight*label+index];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    
      if (active)
      {
      
	for (unsigned int j=0;
	     j<((get_local_size(0)*(i+1)>nPixels) ?
		(nPixels-get_local_size(0)*i) : get_local_size(0));
	     j++)
	{
	  //w = perBlockPosteriors[j]*perBlockPoints[j]*perBlockPoints[j];
	  w = perBlockPosteriors[j];
	  expTerm =							\
	    (currPoint.x-perBlockPoints[j+get_local_size(0)])*(currPoint.x-perBlockPoints[j+get_local_size(0)])+
	    (currPoint.y-perBlockPoints[j+get_local_size(0)*2])*(currPoint.y-perBlockPoints[j+get_local_size(0)*2])+
	    (currPoint.z-perBlockPoints[j])*(currPoint.z-perBlockPoints[j]);
	  expTerm /= (bandwidths[label]*bandwidths[label]);
	  expTerm = native_exp(-expTerm);

	  numTermSum.x += w*expTerm*perBlockPoints[j+get_local_size(0)];
	  numTermSum.y += w*expTerm*perBlockPoints[j+get_local_size(0)*2];
	  numTermSum.z += w*expTerm*perBlockPoints[j];
	  denTermSum += w*expTerm;
	}
      }
    }
  
    if (active)
    {
      m = (numTermSum/denTermSum)-currPoint;
      currPoint += m;
    }

    if (length(m)<MEANSHIFT_THRESHOLD) active = false;
  }

  if (toWrite)
  {
    votes[get_global_id(0)] = currPoint;
  }
}


__kernel void clusterVotes(__global float3 *votes,
			   __global float *weights,
			   float threshold,
			   __global unsigned int *perLabelVotesCount,
			   __local float *perBlockVotes,
			   __local float *perBlockWeights)
{
  unsigned int labelOffset=0, label, nVotes;
  float3 vote;
  float weight=0.0f;
  bool active;
  __local unsigned int temp[N_LABELS];

  if (get_local_id(0)<N_LABELS)
  {
    temp[get_local_id(0)] = perLabelVotesCount[get_local_id(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (label=0; label<N_LABELS; label++)
  {
    labelOffset += temp[label];
    labelOffset += get_local_size(0)-(temp[label]%get_local_size(0));
    if (labelOffset>get_global_id(0))
    {
      labelOffset -= temp[label];
      labelOffset -= get_local_size(0)-(temp[label]%get_local_size(0));
      break;
    }
  }


  nVotes = temp[label];
  active = (get_global_id(0)-labelOffset)<nVotes;
  
  if (active)
  {
      vote = votes[get_global_id(0)];
      weight = weights[get_global_id(0)];
  }

  
  for (unsigned int i=0;
       (get_local_size(0)*i)<nVotes;
       i++)
  {
    if ((get_local_size(0)*i+get_local_id(0))<nVotes)
    {
      //perBlockVotes[get_local_id(0)] = votes[labelOffset+get_local_size(0)*i+get_local_id(0)];
      //perBlockWeights[get_local_id(0)] = weights[labelOffset+get_local_size(0)*i+get_local_id(0)];

      
      perBlockVotes[get_local_id(0)] =					\
        votes[labelOffset+get_local_size(0)*i+get_local_id(0)].x;
      perBlockVotes[get_local_id(0)+get_local_size(0)] =		\
	votes[labelOffset+get_local_size(0)*i+get_local_id(0)].y;
      perBlockVotes[get_local_id(0)+get_local_size(0)*2] =		\
	votes[labelOffset+get_local_size(0)*i+get_local_id(0)].z;
      
      perBlockWeights[get_local_id(0)] = weights[labelOffset+get_local_size(0)*i+get_local_id(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (active)
    {
      for (unsigned int j=0;
	   j<((get_local_size(0)*(i+1)>nVotes) ?
	      (nVotes-get_local_size(0)*i): get_local_size(0));
	   j++)
      {
	//if (length(vote-perBlockVotes[j])<threshold)
	if (((vote.x-perBlockVotes[j])*(vote.x-perBlockVotes[j])+
	     (vote.y-perBlockVotes[j+get_local_size(0)])*(vote.y-perBlockVotes[j+get_local_size(0)])+
	     (vote.z-perBlockVotes[j+get_local_size(0)*2])*(vote.z-perBlockVotes[j+get_local_size(0)*2]))<(threshold*threshold))
	{
	  weight += perBlockWeights[j];
	}
      }
    }
    
  }

  weights[get_global_id(0)] = weight;
}
