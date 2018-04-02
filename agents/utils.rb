def save_screen_record(dir_path)
  images = Dir["#{dir_path}/*"]
  images.each do |image|
    i = Magick::Image.read(image).first
    i = i.resize(RECORD_WIDTH, RECORD_HEIGHT)
    i.write(Pathname(image).sub_ext('.jpg')) do
      self.format='JPEG'
      self.quality=80
    end
  end

  file_name = "#{dir_path}/#{Time.now.to_f * 10000}"
  sequence = ImageList.new(*Dir["#{dir}/*.jpg"].sort)
  sequence.delay = 2
  sequence.ticks_per_second = 60
  sequence.write("#{file_name}.mp4")
  sequence.write("#{file_name}.gif")

  FileUtils.rm_f(Dir["#{dir_path}/*.jpg"])
  FileUtils.rm_f(Dir["#{dir_path}/*.png"])
end