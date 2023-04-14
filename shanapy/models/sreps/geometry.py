import numpy as np

class Geometry:
    @staticmethod
    def sph2cart(rtp):
        """
        Transform spherical to Cartesian coordinates.
        [X,Y,Z] = sph2cart(rthetaphi) transforms corresponding elements of
        data stored in spherical coordinates (azimuth TH, elevation PHI,
        radius R) to Cartesian coordinates X,Y,Z.  The arrays TH, PHI, and
        R must be the same size (or any of them can be scalar).  TH and
        PHI must be in radians.
    
        TH is the counterclockwise angle in the xy plane measured from the
        positive x axis.  PHI is the elevation angle from the xy plane.

        Input rthetaphi:  phi, theta
        Return matrix: n x 3
        """
        if len(rtp.shape) == 2:
            az, elev = rtp[:, 0], rtp[:, 1]
            r = np.ones_like(az)

            z = np.multiply(r, np.sin(elev))[:, np.newaxis]
            rcoselev = np.multiply(r, np.cos(elev))
            x = np.multiply(rcoselev, np.cos(az))[:, np.newaxis]
            y = np.multiply(rcoselev, np.sin(az))[:, np.newaxis]
            return np.hstack((x, y, z))
        else:
            ## input n x k x 2
            n = rtp.shape[0]
            ret = []
            for ni in range(n):
                feat_slice = rtp[ni, :, :]
                feat_cart = Geometry.sph2cart(feat_slice)
                ret.append(feat_cart)
            return np.array(ret)

    @staticmethod
    def cart2sph(xyz):
        """
        Transform Cartesian to spherical coordinates.
        [TH,PHI,R] = cart2sph(X,Y,Z) transforms corresponding elements of
        data stored in Cartesian coordinates X,Y,Z to spherical
        coordinates (azimuth TH, elevation PHI, and radius R).  The arrays
        X,Y, and Z must be the same size (or any of them can be scalar).
        TH and PHI are returned in radians.
    
        TH is the counterclockwise angle in the xy plane measured from the
        positive x axis.  PHI is the elevation angle from the xy plane.

        Input xyz: n x 3 or n x k x 3
        Return n x 3 or n x k x 3
        """
        # x, y, z = xyz
        # XsqPlusYsq = x**2 + y**2
        # r = m.sqrt(XsqPlusYsq + z**2)               # r
        # elev = m.atan2(z,m.sqrt(XsqPlusYsq))
        # az = m.atan2(y,x)

        ## vectorization to speedup
        if len(xyz.shape) == 2:
            xy = xyz[:,0]**2 + xyz[:,1]**2
            r = np.sqrt(xy + xyz[:,2]**2)  #r
            elev = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from Z-axis down
            az = np.arctan2(xyz[:,1], xyz[:,0]) # az (i.e., theta)
            return np.hstack((az[:, np.newaxis], elev[:, np.newaxis], r[:, np.newaxis]))
        else:
            ## input n x k x 3
            n = xyz.shape[0]
            ret = []
            for ni in range(n):
                feat_slice = xyz[ni, :, :]
                feat_sph = Geometry.cart2sph(feat_slice)
                ret.append(feat_sph)
            return np.array(ret)