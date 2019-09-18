
import numpy as np
from shapely.geometry import asPolygon, Polygon

class VideoObjectLocation:
    XMLNamespace = ''

    def __init__(self, visible,frame,abs_time, polygon_points, label=None):
        self.visible = visible
        self.frame = frame
        self.abs_time = abs_time

        self.polygon_points = np.array(polygon_points)   # numpy array to modify polygon points ...
        self.polygon = asPolygon(self.polygon_points)    # Technically a "PolygonAdapter" object

        # for annotated object key-frames
        self.label = label

    def __repr__(self):
        return str(self.polygon)

    def update(self, visible, polygon_points):
        self.visible = visible
        self.polygon_points[:] = polygon_points

    def n_points(self):
        return self.polygon_points.shape[0]

    def intersects(self, other):
        assert isinstance(other, VideoObjectLocation)
        return self.polygon.intersects(other.polygon)

    def area(self):
        return self.polygon.area

    def intersection_area(self, other):
        assert isinstance(other, VideoObjectLocation)
        return self.polygon.intersection(other.polygon).area

    def intersection_percentage(self, other):
        local_area = self.area()
        int_area = self.intersection_area(other)

        return int_area / local_area

    def IOU(self, other):
        local_area = self.area()
        other_area = other.area()
        int_area = self.intersection_area(other)
        union_area = local_area + other_area - int_area

        return int_area / union_area


    def get_XYXY_box(self):
        # return self.x, self.y, self.x + self.w, self.y + self.h
        # TODO: could return a modified version of polygon.bounds, but it would ignore rotation
        raise Exception("VideoObjectLocation: get_XYXY_Box function is now deprecated!")

    @staticmethod
    def fromLocation(original):
        return VideoObjectLocation(original.visible, original.frame, original.abs_time, original.polygon_points,
                                   original.label)

    @staticmethod
    def interpolate(location1, location2, frame):
        assert location1.frame < location2.frame

        if frame <= location1.frame:
            return location1

        if frame >= location2.frame:
            return location2

        # interpolation weight ...
        interval = location2.frame - location1.frame
        w = (frame - location1.frame) / float(interval)
        new_abs_time = location1.abs_time * (1.0 - w) + location2.abs_time * w

        # interpolate the coordinates ....
        new_points = location1.polygon_points * (1.0 - w) + location2.polygon_points * w

        result = VideoObjectLocation(location1.visible, frame, new_abs_time, new_points, location1.label)

        return result

    def toXML(self):
        result = "<VideoObjectLocation>\n"
        result += "  <Visible>" + ("1" if self.visible else "0") + "</Visible>\n"
        result += "  <Frame>" + str(self.frame) + "</Frame>\n"
        result += "  <AbsTime>" + str(self.abs_time) + "</AbsTime>\n"
        if self.label is not None:
            # optional label ...
            result += "  <Label>" + self.label + "</Label>\n"
        result += "  <Polygon>\n"
        for x, y in self.polygon_points:
            result += "    <Point>\n"
            result += "      <X>" + str(x) + "</X>\n"
            result += "      <Y>" + str(y) + "</Y>\n"
            result += "    </Point>\n"

        result += "  </Polygon>\n"

        result += "</VideoObjectLocation>\n"

        return result

    @staticmethod
    def fromXML(root):
        visible = int(root.find(VideoObjectLocation.XMLNamespace + 'Visible').text) > 0
        frame = int(root.find(VideoObjectLocation.XMLNamespace + 'Frame').text)
        abs_time = float(root.find(VideoObjectLocation.XMLNamespace + 'AbsTime').text)

        opt_label_root = root.find(VideoObjectLocation.XMLNamespace + 'Label')
        if opt_label_root is None:
            label = None
        else:
            label = opt_label_root.text

        polygon_root = root.find(VideoObjectLocation.XMLNamespace + 'Polygon')
        if polygon_root is None:
            print("Warning: Legacy Object Location Annotation found")

            # check for old-rectangular based model
            x = float(root.find(VideoObjectLocation.XMLNamespace + 'X').text)
            y = float(root.find(VideoObjectLocation.XMLNamespace + 'Y').text)
            w = float(root.find(VideoObjectLocation.XMLNamespace + 'W').text)
            h = float(root.find(VideoObjectLocation.XMLNamespace + 'H').text)

            polygon_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        else:
            # read polygon
            tempo_points = []
            points_roots = polygon_root.findall(VideoObjectLocation.XMLNamespace + 'Point')
            for point_root in points_roots:
                x = float(point_root.find(VideoObjectLocation.XMLNamespace + 'X').text)
                y = float(point_root.find(VideoObjectLocation.XMLNamespace + 'Y').text)
                tempo_points.append([x, y])

            polygon_points = np.array(tempo_points)

        return VideoObjectLocation(visible, frame, abs_time, polygon_points, label)
