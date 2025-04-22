module Jekyll
  class TagPageGenerator < Generator
    safe true

    def generate(site)
      # Collect all tags from posts
      tags = []
      site.posts.docs.each do |post|
        post.data['tags'].each { |tag| tags << tag } if post.data['tags']
      end
      
      # Remove duplicates
      tags.uniq!
      
      # Create a tag page for each tag
      tags.each do |tag|
        site.pages << TagPage.new(site, site.source, 'tags', tag)
      end
    end
  end

  class TagPage < Page
    def initialize(site, base, dir, tag)
      @site = site
      @base = base
      @dir = dir
      @name = "#{tag.gsub(' ', '-').downcase}.md"

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'tag.html')
      self.data['tag'] = tag
      self.data['title'] = "Posts tagged with #{tag}"
    end
  end
end
