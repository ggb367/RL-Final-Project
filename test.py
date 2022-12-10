        # for i in range(self.num_blocks_x):
        #     for j in range(self.num_blocks_y):
        #         point1 = [x_min + i * self.discritize, y_min + j * self.discritize, self.z]
        #         point2 = [x_min + i * self.discritize + self.discritize, y_min + j * self.discritize, self.z]
        #         point3 = [x_min + i * self.discritize + self.discritize, y_min + j * self.discritize + self.discritize, self.z]
        #         point4 = [x_min + i * self.discritize, y_min + j * self.discritize + self.discritize, self.z]
        #         line1 = [point1, point2]
        #         line2 = [point2, point3]
        #         line3 = [point3, point4]
        #         line4 = [point4, point1]
        #         self.plot_pybullet(lines=[line1, line2, line3, line4])


        # x_min, y_min = self.lower_xy_bounds[0], self.lower_xy_bounds[1]
        # x_max, y_max = self.upper_xy_bounds[0], self.upper_xy_bounds[1]



        #     point1 = [x_min, y_min, self.z]
        # point2 = [x_min, y_max, self.z]
        # point3 = [x_max, y_max, self.z]
        # point4 = [x_max, y_min, self.z]

        # line1 = [point1, point2]
        # line2 = [point2, point3]
        # line3 = [point3, point4]
        # line4 = [point4, point1]
        # self.plot_pybullet(
        #     lines=[line1, line2, line3, line4])
        # # self.sim_scenario.display_gui()

        # print("1111111")