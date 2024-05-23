/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#define MAX_DEPTH 150

#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"


#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

namespace dso {

    class FrameHessian;

    class CalibHessian;

    class FrameShell;


    namespace IOWrap {

        class SampleOutputWrapper : public Output3DWrapper {
        public:
            inline SampleOutputWrapper() {
                numPCL = 0;
                isSavePCL = false;
                isPCLfileClose = false;

                pclFile.open(strTmpFileName);

                printf("OUT: Created SampleOutputWrapper\n");
            }

            virtual ~SampleOutputWrapper() {
                if (pclFile.is_open()) {
                    pclFile.close();
                }
                printf("OUT: Destroyed SampleOutputWrapper\n");
            }

            virtual void publishGraph(const std::map<long, Eigen::Vector2i> &connectivity) {
                printf("OUT: got graph with %d edges\n", (int) connectivity.size());

                int maxWrite = 5;

                for (const std::pair<long, Eigen::Vector2i> &p: connectivity) {
                    int idHost = p.first >> 32;
                    int idTarget = p.first & 0xFFFFFFFF;
                    printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
                    maxWrite--;
                    if (maxWrite == 0) break;
                }
            }


            virtual void publishKeyframes(std::vector<FrameHessian *> &frames, bool final, CalibHessian *HCalib) {
                float fx, fy, cx, cy;
                float fxi, fyi, cxi, cyi;
                //float colorIntensity = 1.0f;
                fx = HCalib->fxl();
                fy = HCalib->fyl();
                cx = HCalib->cxl();
                cy = HCalib->cyl();
                fxi = 1 / fx;
                fyi = 1 / fy;
                cxi = -cx / fx;
                cyi = -cy / fy;
                if (final) {        // only runs at marginalization
                    for (FrameHessian *f: frames) {
                        if (f->shell->poseValid) {
                            auto const &m = f->shell->camToWorld.matrix3x4();

                            printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
                                   f->frameID,
                                   final ? "final" : "non-final",
                                   f->shell->incoming_id,
                                   f->shell->timestamp,
                                   (int) f->pointHessians.size(), (int) f->pointHessiansMarginalized.size(), (int) f->immaturePoints.size());
                            std::cout << f->shell->camToWorld.matrix3x4() << "\n";

                            if (f->shell->poseValid) {
                                Eigen::Vector3f *img = f->dI;
                                float avgDepth;
                                float sumDepth = 0;
                                float validPixel = 0;
                                for (PointHessian *p: f->pointHessiansMarginalized) {
                                    float depth = 1.0f / p->idepth_scaled;
                                    auto const x = (p->u * fxi + cxi) * depth;  // fxi = 1/ fx; cxi = -cx / fx
                                    auto const y = (p->v * fyi + cyi) * depth;
                                    auto const z = depth * (1 + 2 * fxi);
                                    auto const grayColor = img[int(p->v * f->col + p->u)][0];
                                    //printf("(row, col) = (%f, %f) , intensity = %f\n", p->v, p->u, color[0]);

                                    Eigen::Vector4d camPoint(x, y, z, 1.f);
                                    Eigen::Vector3d worldPoint = m * camPoint;

                                    if (z > 0 && z < MAX_DEPTH) {     // MAX_DEPTH
                                        sumDepth += z;
                                        validPixel++;


                                        if (isSavePCL && pclFile.is_open()) {
                                            isWritePCL = true;
                                            // pack r/g/b into rgb for PCL
                                            // https://stackoverflow.com/questions/51501188/unpacking-rgb-values-of-a-point-cloud-from-pcd-file
                                            // https://github.com/PointCloudLibrary/pcl/issues/3293
                                            // https://en.wikipedia.org/wiki/Point_Cloud_Library
                                            // https://stackoverflow.com/questions/63614203/how-to-obtain-color-information-from-a-point-cloud-and-display-it-qt
                                            // https://stackoverflow.com/questions/63614203/how-to-obtain-color-information-from-a-point-cloud-and-display-it-qt
                                            uint32_t rgb = ((uint32_t) grayColor << 16 | (uint32_t) grayColor << 8 | (uint32_t) grayColor);

                                            // x-y-z-gray
                                            pclFile << worldPoint[0] << " " << worldPoint[1] << " " << worldPoint[2] << " " << rgb << "\n";

                                            //                                printf("[%d] Point Cloud Coordinate> X: %.2f, Y: %.2f, Z: %.2f\n",
                                            //                                         numPCL,
                                            //                                         worldPoint[0],
                                            //                                         worldPoint[1],
                                            //                                         worldPoint[2]);

                                            numPCL++;
                                            isWritePCL = false;
                                        } else {
                                            if (!isPCLfileClose) {
                                                if (pclFile.is_open()) {
                                                    pclFile.flush();
                                                    pclFile.close();
                                                    isPCLfileClose = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                // write depth value
//                                if (validPixel > 0) {
//                                    avgDepth = sumDepth / validPixel;
//                                    std::ofstream file;
//                                    file.open("avg_frame_depth.txt", std::ios::app | std::ios::out);
//                                    if (file.is_open())
//                                        file << f->shell->id << " " << avgDepth << std::endl;
//                                    file.close();
//                                }
                            }
                        }
                    }
                }
            }

            virtual void publishCamPose(FrameShell *frame, CalibHessian *HCalib) {
                printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
                       frame->incoming_id,
                       frame->timestamp,
                       frame->id);
                std::cout << frame->camToWorld.matrix3x4() << "\n";
            }


            virtual void pushLiveFrame(FrameHessian *image) {
                // can be used to get the raw image / intensity pyramid.
            }

            virtual void pushDepthImage(MinimalImageB3 *image) {
                // can be used to get the raw image with depth overlay.
            }

            virtual bool needPushDepthImage() {
                return false;
            }

            virtual void pushDepthImageFloat(MinimalImageF *image, FrameHessian *KF) {
                printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
                       KF->frameID,
                       KF->shell->incoming_id,
                       KF->shell->timestamp,
                       KF->shell->id);
                std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

                int maxWrite = 5;
                for (int y = 0; y < image->h; y++) {
                    for (int x = 0; x < image->w; x++) {
                        if (image->at(x, y) <= 0) continue;

                        printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x, y, image->at(x, y));
                        maxWrite--;
                        if (maxWrite == 0) break;
                    }
                    if (maxWrite == 0) break;
                }
            }

            std::ofstream pclFile;

        };


    }


}
