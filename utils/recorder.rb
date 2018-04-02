# Class Recorder
class Recorder
  def self.save_screen_png(env, step, interval)
    filename = "./results/#{TIME_STAMP}/#{(Time.now.to_f * 10_000).to_i}.png"
    env.save_screen_PNG(filename) if step % interval == interval - 1
  end

  def self.save_screen_record(dir_path)
    convert_images_from_png_to_jpg(dir_path)
    file_name = "#{dir_path}/#{Time.now.to_f * 10_000}"
    sequence = Magick::ImageList.new(*Dir["#{dir_path}/*.jpg"].sort)
    sequence.delay = 2
    sequence.ticks_per_second = 60
    sequence.write("#{file_name}.mp4")
    sequence.write("#{file_name}.gif")

    FileUtils.rm_f(Dir["#{dir_path}/*.jpg"])
    FileUtils.rm_f(Dir["#{dir_path}/*.png"])
  end

  private_class_method

  def self.convert_images_from_png_to_jpg(dir_path)
    images = Dir["#{dir_path}/*"]
    images.each do |image|
      i = Magick::Image.read(image).first
      i = i.resize(RECORD_WIDTH, RECORD_HEIGHT)
      i.write(Pathname(image).sub_ext('.jpg')) do
        self.format = 'JPEG'
        self.quality = 80
      end
    end
  end
end
